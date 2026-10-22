//! Integer trace certificates for a complete causal scan.
use super::*;
use crate::lookup::table::{Table, TableType};
use anyhow::Context;
use std::ops::Range;

pub(super) const S: usize = 0;
pub(super) const U: usize = 1;
pub(super) const RDEC: usize = 2;
pub(super) const RUPD: usize = 3;
pub(super) const M: usize = 4;
pub(super) const B: usize = 5;
pub(super) const D: usize = 6;
pub(super) const O: usize = 7;
pub(super) const RMEM: usize = 8;
pub(super) const RVAL: usize = 9;
pub(super) const RDELTA: usize = 10;
pub(super) const ROUT: usize = 11;
pub(super) const GROUPS: usize = 12;

#[derive(Clone, Debug)]
pub(super) struct Column {
    pub bits: usize,
    pub state: bool,
}
impl Column {
    pub fn table(&self) -> Table {
        Table::new(
            TableType::ShiftCheck,
            1f32.to_bits(),
            1f32.to_bits(),
            self.bits,
        )
    }
}
#[derive(Clone, Debug)]
pub(super) struct Group {
    pub columns: Range<usize>,
    pub signed: bool,
    pub state: bool,
}
#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub columns: Vec<Column>,
    pub groups: Vec<Group>,
}
impl Layout {
    pub fn new(quant: &DeltaQuant) -> Self {
        let mut columns = Vec::new();
        let mut groups = Vec::new();
        for (bits, signed, state) in [
            (32, true, true),
            (32, true, true),
            (quant.decay.shift, false, true),
            (quant.update.shift, false, true),
            (32, true, false),
            (32, true, false),
            (32, true, false),
            (32, true, false),
            (quant.memory.shift, false, false),
            (quant.value.shift, false, false),
            (quant.delta.shift, false, false),
            (quant.output.shift, false, false),
        ] {
            let start = columns.len();
            let mut remaining = bits as usize;
            while remaining > 0 {
                // LogUp requires tables with at least four entries. Avoid a
                // one-bit final limb by using 15+2 bits instead of 16+1.
                let width = if remaining == 17 {
                    15
                } else {
                    remaining.min(16)
                };
                columns.push(Column { bits: width, state });
                remaining -= width;
            }
            groups.push(Group {
                columns: start..columns.len(),
                signed,
                state,
            });
        }
        Self { columns, groups }
    }
    pub fn tables(&self) -> Vec<Table> {
        self.columns
            .iter()
            .map(Column::table)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    pub fn put<F: PrimeField>(
        &self,
        columns: &mut [Vec<F>],
        group: usize,
        index: usize,
        value: i128,
    ) -> Result<()> {
        let group = &self.groups[group];
        let value = if group.signed {
            bounded(i64::try_from(value)?)?;
            (value + (1i128 << 31)) as u128
        } else {
            ensure!(value >= 0, "Negative DeltaNet division remainder");
            value as u128
        };
        let bits: usize = self.columns[group.columns.clone()]
            .iter()
            .map(|c| c.bits)
            .sum();
        ensure!(
            bits < 128 && value < (1u128 << bits),
            "DeltaNet certificate exceeds its limb range"
        );
        let mut offset = 0;
        for column in group.columns.clone() {
            columns[column][index] =
                F::from(((value >> offset) & ((1u128 << self.columns[column].bits) - 1)) as u64);
            offset += self.columns[column].bits;
        }
        Ok(())
    }
    pub fn reconstruct<F: PrimeField>(&self, group: usize, values: &[F]) -> F {
        let group = &self.groups[group];
        let value = group.columns.clone().rev().fold(F::ZERO, |a, i| {
            a * F::from(1u64 << self.columns[i].bits) + values[i]
        });
        value
            - if group.signed {
                F::from(1u64 << 31)
            } else {
                F::ZERO
            }
    }
}

/// Build all columns without holding another copy of the sequence-wide states.
/// The initial state is fixed to zero; cached inference is never a proof witness.
pub(super) fn build<F: PrimeField>(
    layer: &GatedDeltaScan<Element>,
    data: &[Vec<Element>],
    rows: usize,
    valid: usize,
) -> Result<Vec<Vec<F>>> {
    let quant = layer.quant.as_ref().context("DeltaNet is not quantized")?;
    let layout = Layout::new(quant);
    let vector_len = rows * layer.heads * layer.value_dim;
    let state_len = vector_len * layer.key_dim;
    ensure!(
        rows.is_power_of_two() && vector_len >= 4 && valid <= rows,
        "Invalid DeltaNet proof shape"
    );
    let mut columns: Vec<Vec<F>> = layout
        .columns
        .iter()
        .map(|c| vec![F::ZERO; if c.state { state_len } else { vector_len }])
        .collect();
    // Offset-binary signed limbs also encode zero on all padded rows.
    for group in layout.groups.iter().filter(|g| g.signed) {
        columns[group.columns.end - 1].fill(F::from(32768u64));
    }
    ensure!(
        quant.input_domains.len() == 5,
        "Missing DeltaNet input domains"
    );
    for (values, &(minimum, maximum)) in data.iter().zip(&quant.input_domains) {
        ensure!(
            values.iter().all(|x| *x >= minimum && *x <= maximum),
            "DeltaNet witness input outside proved domain"
        );
    }
    let hsize = layer.key_dim * layer.value_dim;
    let mut states = vec![0i64; layer.heads * hsize];
    let mut memory = vec![0i128; layer.value_dim];
    let mut delta = vec![0i64; layer.value_dim];
    for t in 0..valid {
        for h in 0..layer.heads {
            let s = &mut states[h * hsize..][..hsize];
            let q = &data[0][(t * layer.heads + h) * layer.key_dim..][..layer.key_dim];
            let k = &data[1][(t * layer.heads + h) * layer.key_dim..][..layer.key_dim];
            let v = &data[2][(t * layer.heads + h) * layer.value_dim..][..layer.value_dim];
            let alpha = data[3][t * layer.heads + h];
            let beta = data[4][t * layer.heads + h];
            let base = (t * layer.heads + h) * hsize;
            for (index, x) in s.iter_mut().enumerate() {
                let (decayed, remainder) = quant
                    .decay
                    .apply_with_remainder(*x as i128 * alpha as i128)?;
                *x = bounded(decayed)?;
                layout.put(&mut columns, U, base + index, *x as i128)?;
                layout.put(&mut columns, RDEC, base + index, remainder as i128)?;
            }
            memory.fill(0);
            for i in 0..layer.key_dim {
                for j in 0..layer.value_dim {
                    memory[j] += k[i] as i128 * s[i * layer.value_dim + j] as i128;
                }
            }
            for j in 0..layer.value_dim {
                let index = (t * layer.heads + h) * layer.value_dim + j;
                let (m, rm) = quant.memory.apply_with_remainder(memory[j])?;
                let (b, rb) = quant.value.apply_with_remainder(v[j] as i128)?;
                let (d, rd) = quant
                    .delta
                    .apply_with_remainder((b as i128 - m as i128) * beta as i128)?;
                delta[j] = bounded(d)?;
                for (group, value) in [
                    (M, m as i128),
                    (B, b as i128),
                    (D, d as i128),
                    (RMEM, rm as i128),
                    (RVAL, rb as i128),
                    (RDELTA, rd as i128),
                ] {
                    layout.put(&mut columns, group, index, value)?;
                }
            }
            for i in 0..layer.key_dim {
                for j in 0..layer.value_dim {
                    let index = i * layer.value_dim + j;
                    let (update, remainder) = quant
                        .update
                        .apply_with_remainder(k[i] as i128 * delta[j] as i128)?;
                    s[index] = bounded(
                        s[index]
                            .checked_add(update)
                            .context("DeltaNet witness state overflow")?,
                    )?;
                    layout.put(&mut columns, S, base + index, s[index] as i128)?;
                    layout.put(&mut columns, RUPD, base + index, remainder as i128)?;
                }
            }
            memory.fill(0);
            for i in 0..layer.key_dim {
                for j in 0..layer.value_dim {
                    memory[j] += q[i] as i128 * s[i * layer.value_dim + j] as i128;
                }
            }
            for j in 0..layer.value_dim {
                let index = (t * layer.heads + h) * layer.value_dim + j;
                let (output, remainder) = quant.output.apply_with_remainder(memory[j])?;
                layout.put(&mut columns, O, index, output as i128)?;
                layout.put(&mut columns, ROUT, index, remainder as i128)?;
            }
        }
    }
    Ok(columns)
}
