//! Length-limited Huffman codes and their run-length coded description (libwebp's
//! `utils/huffman_encode_utils.c`).

const MAX_ALLOWED_CODE_LENGTH: usize = 15;

#[derive(Clone, Copy, Default)]
pub(crate) struct HuffmanTreeToken {
    pub(crate) code: u8,
    pub(crate) extra_bits: u8,
}

#[derive(Clone, Default)]
pub(crate) struct HuffmanTreeCode {
    pub(crate) code_lengths: Vec<u8>,
    pub(crate) codes: Vec<u16>,
}

impl HuffmanTreeCode {
    pub(crate) fn num_symbols(&self) -> usize {
        self.code_lengths.len()
    }
}

#[derive(Clone, Copy, Default)]
struct HuffmanTree {
    total_count: u32,
    value: i32,
    pool_index_left: i32,
    pool_index_right: i32,
}

fn values_should_be_collapsed_to_stride_average(a: i64, b: i64) -> bool {
    (a - b).abs() < 4
}

fn optimize_huffman_for_rle(length: usize, good_for_rle: &mut [u8], counts: &mut [u32]) {
    let mut length = length as isize;
    while length >= 0 {
        if length == 0 {
            return;
        }
        if counts[length as usize - 1] != 0 {
            break;
        }
        length -= 1;
    }
    let length = length as usize;
    {
        let mut symbol = counts[0];
        let mut stride = 0usize;
        for i in 0..length + 1 {
            if i == length || counts[i] != symbol {
                if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
                    for k in 0..stride {
                        good_for_rle[i - k - 1] = 1;
                    }
                }
                stride = 1;
                if i != length {
                    symbol = counts[i];
                }
            } else {
                stride += 1;
            }
        }
    }
    {
        let mut stride = 0u32;
        let mut limit = counts[0];
        let mut sum = 0u32;
        for i in 0..length + 1 {
            if i == length
                || good_for_rle[i] != 0
                || (i != 0 && good_for_rle[i - 1] != 0)
                || !values_should_be_collapsed_to_stride_average(
                    i64::from(counts[i]),
                    i64::from(limit),
                )
            {
                if stride >= 4 || (stride >= 3 && sum == 0) {
                    let mut count = (sum + stride / 2) / stride;
                    if count < 1 {
                        count = 1;
                    }
                    if sum == 0 {
                        count = 0;
                    }
                    for k in 0..stride as usize {
                        counts[i - k - 1] = count;
                    }
                }
                stride = 0;
                sum = 0;
                if (i as isize) < length as isize - 3 {
                    limit = (counts[i] + counts[i + 1] + counts[i + 2] + counts[i + 3] + 2) / 4;
                } else if i < length {
                    limit = counts[i];
                } else {
                    limit = 0;
                }
            }
            stride += 1;
            if i != length {
                sum += counts[i];
                if stride >= 4 {
                    limit = (sum + stride / 2) / stride;
                }
            }
        }
    }
}

fn set_bit_depths(tree: &HuffmanTree, pool: &[HuffmanTree], bit_depths: &mut [u8], level: u8) {
    if tree.pool_index_left >= 0 {
        set_bit_depths(
            &pool[tree.pool_index_left as usize],
            pool,
            bit_depths,
            level + 1,
        );
        set_bit_depths(
            &pool[tree.pool_index_right as usize],
            pool,
            bit_depths,
            level + 1,
        );
    } else {
        bit_depths[tree.value as usize] = level;
    }
}

fn generate_optimal_tree(
    histogram: &[u32],
    tree: &mut [HuffmanTree],
    tree_depth_limit: u8,
    bit_depths: &mut [u8],
) {
    let histogram_size = histogram.len();
    let tree_size_orig = histogram.iter().filter(|&&h| h != 0).count();
    if tree_size_orig == 0 {
        return;
    }
    let (tree, tree_pool) = tree.split_at_mut(tree_size_orig);
    let mut count_min = 1u32;
    loop {
        let mut tree_size = tree_size_orig;
        let mut idx = 0;
        for (j, &h) in histogram.iter().enumerate() {
            if h != 0 {
                let count = if h < count_min { count_min } else { h };
                tree[idx] = HuffmanTree {
                    total_count: count,
                    value: j as i32,
                    pool_index_left: -1,
                    pool_index_right: -1,
                };
                idx += 1;
            }
        }
        tree[..tree_size].sort_unstable_by(|t1, t2| {
            t2.total_count
                .cmp(&t1.total_count)
                .then(t1.value.cmp(&t2.value))
        });
        if tree_size > 1 {
            let mut tree_pool_size = 0usize;
            while tree_size > 1 {
                tree_pool[tree_pool_size] = tree[tree_size - 1];
                tree_pool[tree_pool_size + 1] = tree[tree_size - 2];
                tree_pool_size += 2;
                let count = tree_pool[tree_pool_size - 1]
                    .total_count
                    .wrapping_add(tree_pool[tree_pool_size - 2].total_count);
                tree_size -= 2;
                let k = tree[..tree_size]
                    .iter()
                    .position(|t| t.total_count <= count)
                    .unwrap_or(tree_size);
                tree.copy_within(k..tree_size, k + 1);
                tree[k] = HuffmanTree {
                    total_count: count,
                    value: -1,
                    pool_index_left: tree_pool_size as i32 - 1,
                    pool_index_right: tree_pool_size as i32 - 2,
                };
                tree_size += 1;
            }
            set_bit_depths(&tree[0], tree_pool, bit_depths, 0);
        } else if tree_size == 1 {
            bit_depths[tree[0].value as usize] = 1;
        }
        let max_depth = bit_depths[..histogram_size]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        if max_depth <= tree_depth_limit {
            break;
        }
        count_min = count_min.wrapping_mul(2);
    }
}

fn code_repeated_values(
    mut repetitions: usize,
    tokens: &mut Vec<HuffmanTreeToken>,
    value: u8,
    prev_value: u8,
) {
    if value != prev_value {
        tokens.push(HuffmanTreeToken {
            code: value,
            extra_bits: 0,
        });
        repetitions -= 1;
    }
    while repetitions >= 1 {
        if repetitions < 3 {
            for _ in 0..repetitions {
                tokens.push(HuffmanTreeToken {
                    code: value,
                    extra_bits: 0,
                });
            }
            break;
        } else if repetitions < 7 {
            tokens.push(HuffmanTreeToken {
                code: 16,
                extra_bits: (repetitions - 3) as u8,
            });
            break;
        } else {
            tokens.push(HuffmanTreeToken {
                code: 16,
                extra_bits: 3,
            });
            repetitions -= 6;
        }
    }
}

fn code_repeated_zeros(mut repetitions: usize, tokens: &mut Vec<HuffmanTreeToken>) {
    while repetitions >= 1 {
        if repetitions < 3 {
            for _ in 0..repetitions {
                tokens.push(HuffmanTreeToken {
                    code: 0,
                    extra_bits: 0,
                });
            }
            break;
        } else if repetitions < 11 {
            tokens.push(HuffmanTreeToken {
                code: 17,
                extra_bits: (repetitions - 3) as u8,
            });
            break;
        } else if repetitions < 139 {
            tokens.push(HuffmanTreeToken {
                code: 18,
                extra_bits: (repetitions - 11) as u8,
            });
            break;
        } else {
            tokens.push(HuffmanTreeToken {
                code: 18,
                extra_bits: 0x7f,
            });
            repetitions -= 138;
        }
    }
}

pub(crate) fn create_compressed_huffman_tree(
    tree: &HuffmanTreeCode,
    tokens: &mut Vec<HuffmanTreeToken>,
) {
    tokens.clear();
    let depth_size = tree.num_symbols();
    let mut prev_value = 8u8;
    let mut i = 0;
    while i < depth_size {
        let value = tree.code_lengths[i];
        let mut k = i + 1;
        while k < depth_size && tree.code_lengths[k] == value {
            k += 1;
        }
        let runs = k - i;
        if value == 0 {
            code_repeated_zeros(runs, tokens);
        } else {
            code_repeated_values(runs, tokens, value, prev_value);
            prev_value = value;
        }
        i += runs;
    }
}

const REVERSED_BITS: [u8; 16] = [
    0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
];

fn reverse_bits(num_bits: u32, mut bits: u32) -> u32 {
    let mut retval = 0u32;
    let mut i = 0;
    while i < num_bits {
        i += 4;
        retval |= u32::from(REVERSED_BITS[(bits & 0xf) as usize])
            << (MAX_ALLOWED_CODE_LENGTH as u32 + 1 - i);
        bits >>= 4;
    }
    retval >> (MAX_ALLOWED_CODE_LENGTH as u32 + 1 - num_bits)
}

fn convert_bit_depths_to_symbols(tree: &mut HuffmanTreeCode) {
    let mut next_code = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
    let mut depth_count = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
    for &l in &tree.code_lengths {
        depth_count[l as usize] += 1;
    }
    depth_count[0] = 0;
    let mut code = 0u32;
    for i in 1..=MAX_ALLOWED_CODE_LENGTH {
        code = (code + depth_count[i - 1]) << 1;
        next_code[i] = code;
    }
    for i in 0..tree.code_lengths.len() {
        let code_length = tree.code_lengths[i] as usize;
        tree.codes[i] = reverse_bits(code_length as u32, next_code[code_length]) as u16;
        next_code[code_length] = next_code[code_length].wrapping_add(1);
    }
}

/// Builds the code of `histogram` (which is modified for better run-length coding) into
/// `code`, sized to the histogram, with no length above `tree_depth_limit`.
pub(crate) fn create_huffman_tree(
    histogram: &mut [u32],
    tree_depth_limit: u8,
    code: &mut HuffmanTreeCode,
) {
    let num_symbols = histogram.len();
    let mut buf_rle = vec![0u8; num_symbols];
    optimize_huffman_for_rle(num_symbols, &mut buf_rle, histogram);
    code.code_lengths.clear();
    code.code_lengths.resize(num_symbols, 0);
    code.codes.clear();
    code.codes.resize(num_symbols, 0);
    let mut tree = vec![HuffmanTree::default(); 3 * num_symbols];
    generate_optimal_tree(
        histogram,
        &mut tree,
        tree_depth_limit,
        &mut code.code_lengths,
    );
    convert_bit_depths_to_symbols(code);
}
