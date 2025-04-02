s[PaddedInput].compute_line()

## Cache Tensor Core
out_wmma_accumulator = s.cache_write(out, wmma.accumulator)

## Cache read shared
PaddedInput_shared = s.cache_read(PaddedInput, shared, out.wmma.accumulator)
filter_shared = s.cache_read(filter, shared, out.wmma.accumulator)
PaddedInput_shared_wmma_matrix_a = s.cache_read(PaddedInput_shared, wmma.matrix_a, out.wmma.accumulator)
filter_shared_wmma_matrix_b = s.cache_read(filter_shared, wmma.matrix_b, out.wmma.accumulator)

## Cache read shared
out_wmma_accumulator_shared = s.cache_read(out_wmma_accumulator, shared, out)

#==--------- Start schedule STAGE out ----------==#

## Unroll pragma 
nn_o, nn_i = s[out].split(nn, nparts = 1)
yy_o, yy_i = s[out].split(yy, nparts = 1)
xx_o, xx_i = s[out].split(xx, nparts = 1)
ff_o, ff_i = s[out].split(ff, nparts = 1)
s[out].reorder(nn_o, yy_o, xx_o, ff_o, nn_i, yy_i, xx_i, ff_i, )

## Bind blockIdx.x

## tile spatial 
nn_i_o, nn_i_i = s[out].split(nn_i, nparts = 1)
yy_i_o, yy_i_i = s[out].split(yy_i, nparts = 1)
xx_i_o, xx_i_i = s[out].split(xx_i, nparts = 1)
ff_i_o, ff_i_i = s[out].split(ff_i, nparts = 1)
s[out].reorder(nn_i_o, yy_i_o, xx_i_o, ff_i_o, nn_i_i, yy_i_i, xx_i_i, ff_i_i, )
nn_i_o_yy_i_o_f_xx_i_o_f_ff_i_o_f = s[out].fuse(nn_i_o, yy_i_o, xx_i_o, ff_i_o, )
s[out].bind(nn_i_o_yy_i_o_f_xx_i_o_f_ff_i_o_f, te.thread_axis("blockIdx.x"))

## Bind threadIdx.y

## tile spatial 
nn_i_i_o, nn_i_i_i = s[out].split(nn_i_i, nparts = 1)
yy_i_i_o, yy_i_i_i = s[out].split(yy_i_i, nparts = 1)
xx_i_i_o, xx_i_i_i = s[out].split(xx_i_i, nparts = 1)
ff_i_i_o, ff_i_i_i = s[out].split(ff_i_i, nparts = 1)
s[out].reorder(nn_i_i_o, yy_i_i_o, xx_i_i_o, ff_i_i_o, nn_i_i_i, yy_i_i_i, xx_i_i_i, ff_i_i_i, )
nn_i_i_o_yy_i_i_o_f_xx_i_i_o_f_ff_i_i_o_f = s[out].fuse(nn_i_i_o, yy_i_i_o, xx_i_i_o, ff_i_i_o, )
s[out].bind(nn_i_i_o_yy_i_i_o_f_xx_i_i_o_f_ff_i_i_o_f, te.thread_axis("threadIdx.y"))

## Bind threadIdx.x

## tile spatial 
nn_i_i_i_o, nn_i_i_i_i = s[out].split(nn_i_i_i, nparts = 1)
yy_i_i_i_o, yy_i_i_i_i = s[out].split(yy_i_i_i, nparts = 1)
xx_i_i_i_o, xx_i_i_i_i = s[out].split(xx_i_i_i, nparts = 1)
ff_i_i_i_o, ff_i_i_i_i = s[out].split(ff_i_i_i, nparts = 1)
s[out].reorder(nn_i_i_i_o, yy_i_i_i_o, xx_i_i_i_o, ff_i_i_i_o, nn_i_i_i_i, yy_i_i_i_i, xx_i_i_i_i, ff_i_i_i_i, )
nn_i_i_i_o_yy_i_i_i_o_f_xx_i_i_i_o_f_ff_i_i_i_o_f = s[out].fuse(nn_i_i_i_o, yy_i_i_i_o, xx_i_i_i_o, ff_i_i_i_o, )
s[out].bind(nn_i_i_i_o_yy_i_i_i_o_f_xx_i_i_i_o_f_ff_i_i_i_o_f, te.thread_axis("threadIdx.x"))

## Vectorize 

## tile spatial 
nn_i_i_i_i_o, nn_i_i_i_i_i = s[out].split(nn_i_i_i_i, nparts = 1)
yy_i_i_i_i_o, yy_i_i_i_i_i = s[out].split(yy_i_i_i_i, nparts = 1)
xx_i_i_i_i_o, xx_i_i_i_i_i = s[out].split(xx_i_i_i_i, nparts = 1)
ff_i_i_i_i_o, ff_i_i_i_i_i = s[out].split(ff_i_i_i_i, nparts = 1)
s[out].reorder(nn_i_i_i_i_o, yy_i_i_i_i_o, xx_i_i_i_i_o, ff_i_i_i_i_o, nn_i_i_i_i_i, yy_i_i_i_i_i, xx_i_i_i_i_i, ff_i_i_i_i_i, )
nn_i_i_i_i_i_yy_i_i_i_i_i_f_xx_i_i_i_i_i_f_ff_i_i_i_i_i_f = s[out].fuse(nn_i_i_i_i_i, yy_i_i_i_i_i, xx_i_i_i_i_i, ff_i_i_i_i_i, )
s[out].vectorize(nn_i_i_i_i_i_yy_i_i_i_i_i_f_xx_i_i_i_i_i_f_ff_i_i_i_i_i_f)

# Var nn_o length 1
# Var yy_o length 1
# Var xx_o length 1
# Var ff_o length 1
# Var nn_i_o_yy_i_o_f_xx_i_o_f_ff_i_o_f length 1
# Var nn_i_i_o_yy_i_i_o_f_xx_i_i_o_f_ff_i_i_o_f length 1
# Var nn_i_i_i_o_yy_i_i_i_o_f_xx_i_i_i_o_f_ff_i_i_i_o_f length 1
# Var nn_i_i_i_i_o length 1
# Var yy_i_i_i_i_o length 1
# Var xx_i_i_i_i_o length 1
# Var ff_i_i_i_i_o length 1
# Var nn_i_i_i_i_i_yy_i_i_i_i_i_f_xx_i_i_i_i_i_f_ff_i_i_i_i_i_f length 1
#==--------- Start schedule STAGE out.wmma.accumulator.shared ----------==#
s[out_wmma_accumulator_shared].compute_at(s[out], ff_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
# Var ax3 length 1
## Storage align 
s[out_wmma_accumulator_shared].storage_align(ax2, 0.000000, 1.000000)

## Bind threadIdx.y

## tile spatial 
ax0_o, ax0_i = s[out_wmma_accumulator_shared].split(ax0, nparts = 1)
ax1_o, ax1_i = s[out_wmma_accumulator_shared].split(ax1, nparts = 1)
ax2_o, ax2_i = s[out_wmma_accumulator_shared].split(ax2, nparts = 1)
ax3_o, ax3_i = s[out_wmma_accumulator_shared].split(ax3, nparts = 1)
s[out_wmma_accumulator_shared].reorder(ax0_o, ax1_o, ax2_o, ax3_o, ax0_i, ax1_i, ax2_i, ax3_i, )
ax0_o_ax1_o_f_ax2_o_f_ax3_o_f = s[out_wmma_accumulator_shared].fuse(ax0_o, ax1_o, ax2_o, ax3_o, )
s[out_wmma_accumulator_shared].bind(ax0_o_ax1_o_f_ax2_o_f_ax3_o_f, te.thread_axis("threadIdx.y"))

## Tensor core store
ax0_i_o, ax0_i_i = s[out_wmma_accumulator_shared].split(ax0_i, factor = 16)
ax1_i_o, ax1_i_i = s[out_wmma_accumulator_shared].split(ax1_i, factor = 1)
ax2_i_o, ax2_i_i = s[out_wmma_accumulator_shared].split(ax2_i, factor = 1)
ax3_i_o, ax3_i_i = s[out_wmma_accumulator_shared].split(ax3_i, factor = 16)
s[out_wmma_accumulator_shared].reorder(ax0_i_o, ax1_i_o, ax2_i_o, ax3_i_o, ax0_i_i, ax1_i_i, ax2_i_i, ax3_i_i, )
s[out_wmma_accumulator_shared].tensorize(ax0_i_i, intrin_wmma_store_matrix(
[sc_n0, sc_n1, sc_n2, 1], [lc_n0, lc_n1, lc_n2, 1], (16, 16, 16), float16, [16, 1, 1, 16], [16, 1, 1, 16], 
))

# Var ax0_o_ax1_o_f_ax2_o_f_ax3_o_f length 1
# Var ax0_i_o length 1
# Var ax1_i_o length 1
# Var ax2_i_o length 1
# Var ax3_i_o length 1
# Var ax0_i_i length 1
# Var ax1_i_i length 1
# Var ax2_i_i length 1
# Var ax3_i_i length 1
#==--------- Start schedule STAGE out.wmma.accumulator ----------==#
s[out_wmma_accumulator].compute_at(s[out_wmma_accumulator_shared], ax0_o_ax1_o_f_ax2_o_f_ax3_o_f)

# Var nn_c length 1
# Var yy_c length 1
# Var xx_c length 1
# Var ff_c length 1
# Var ry
# Var rx
# Var rc
## general tile 

## tile 
nn_c_o, nn_c_i = s[out_wmma_accumulator].split(nn_c, nparts = 1)
yy_c_o, yy_c_i = s[out_wmma_accumulator].split(yy_c, nparts = 1)
xx_c_o, xx_c_i = s[out_wmma_accumulator].split(xx_c, nparts = 1)
ff_c_o, ff_c_i = s[out_wmma_accumulator].split(ff_c, nparts = 1)
ry_o, ry_i = s[out_wmma_accumulator].split(ry, nparts = 1)
rx_o, rx_i = s[out_wmma_accumulator].split(rx, nparts = 1)
rc_o, rc_i = s[out_wmma_accumulator].split(rc, nparts = 1)
s[out_wmma_accumulator].reorder(nn_c_o, yy_c_o, xx_c_o, ff_c_o, ry_o, rx_o, rc_o, nn_c_i, yy_c_i, xx_c_i, ff_c_i, ry_i, rx_i, rc_i, )

## tile 
nn_c_i_o, nn_c_i_i = s[out_wmma_accumulator].split(nn_c_i, nparts = 1)
yy_c_i_o, yy_c_i_i = s[out_wmma_accumulator].split(yy_c_i, nparts = 1)
xx_c_i_o, xx_c_i_i = s[out_wmma_accumulator].split(xx_c_i, nparts = 1)
ff_c_i_o, ff_c_i_i = s[out_wmma_accumulator].split(ff_c_i, nparts = 1)
ry_i_o, ry_i_i = s[out_wmma_accumulator].split(ry_i, nparts = 1)
rx_i_o, rx_i_i = s[out_wmma_accumulator].split(rx_i, nparts = 1)
rc_i_o, rc_i_i = s[out_wmma_accumulator].split(rc_i, nparts = 1)
s[out_wmma_accumulator].reorder(nn_c_i_o, yy_c_i_o, xx_c_i_o, ff_c_i_o, ry_i_o, rx_i_o, rc_i_o, nn_c_i_i, yy_c_i_i, xx_c_i_i, ff_c_i_i, ry_i_i, rx_i_i, rc_i_i, )

## tile 
nn_c_i_i_o, nn_c_i_i_i = s[out_wmma_accumulator].split(nn_c_i_i, nparts = 1)
yy_c_i_i_o, yy_c_i_i_i = s[out_wmma_accumulator].split(yy_c_i_i, nparts = 1)
xx_c_i_i_o, xx_c_i_i_i = s[out_wmma_accumulator].split(xx_c_i_i, nparts = 1)
ff_c_i_i_o, ff_c_i_i_i = s[out_wmma_accumulator].split(ff_c_i_i, nparts = 1)
ry_i_i_o, ry_i_i_i = s[out_wmma_accumulator].split(ry_i_i, nparts = 1)
rx_i_i_o, rx_i_i_i = s[out_wmma_accumulator].split(rx_i_i, nparts = 1)
rc_i_i_o, rc_i_i_i = s[out_wmma_accumulator].split(rc_i_i, nparts = 1)
s[out_wmma_accumulator].reorder(nn_c_i_i_o, yy_c_i_i_o, xx_c_i_i_o, ff_c_i_i_o, ry_i_i_o, rx_i_i_o, rc_i_i_o, nn_c_i_i_i, yy_c_i_i_i, xx_c_i_i_i, ff_c_i_i_i, ry_i_i_i, rx_i_i_i, rc_i_i_i, )

## Tensor core compute
nn_c_i_i_i_o, nn_c_i_i_i_i = s[out_wmma_accumulator].split(nn_c_i_i_i, factor = 16)
yy_c_i_i_i_o, yy_c_i_i_i_i = s[out_wmma_accumulator].split(yy_c_i_i_i, factor = 1)
xx_c_i_i_i_o, xx_c_i_i_i_i = s[out_wmma_accumulator].split(xx_c_i_i_i, factor = 1)
ff_c_i_i_i_o, ff_c_i_i_i_i = s[out_wmma_accumulator].split(ff_c_i_i_i, factor = 16)
ry_i_i_i_o, ry_i_i_i_i = s[out_wmma_accumulator].split(ry_i_i_i, factor = 1)
rx_i_i_i_o, rx_i_i_i_i = s[out_wmma_accumulator].split(rx_i_i_i, factor = 1)
rc_i_i_i_o, rc_i_i_i_i = s[out_wmma_accumulator].split(rc_i_i_i, factor = 16)
s[out_wmma_accumulator].reorder(nn_c_i_i_i_o, yy_c_i_i_i_o, xx_c_i_i_i_o, ff_c_i_i_i_o, ry_i_i_i_o, rx_i_i_i_o, rc_i_i_i_o, nn_c_i_i_i_i, yy_c_i_i_i_i, xx_c_i_i_i_i, ff_c_i_i_i_i, ry_i_i_i_i, rx_i_i_i_i, rc_i_i_i_i, )
s[out_wmma_accumulator].tensorize(nn_c_i_i_i_i, intrin_wmma_gemm(
Tensor(shape=[16, 1, 1, 16], op.name=A), Tensor(shape=[16, 16], op.name=B), Tensor(shape=[16, 1, 1, 16], op.name=C), [la_k0, la_k1, la_k2, 1], [lb_n0, 1], [lc_n0, lc_n1, lc_n2, 1], (16, 16, 16), 
))

# Var nn_c_o length 1
# Var yy_c_o length 1
# Var xx_c_o length 1
# Var ff_c_o length 1
# Var ry_o length 1
# Var rx_o length 1
# Var rc_o length 1
# Var nn_c_i_o length 1
# Var yy_c_i_o length 1
# Var xx_c_i_o length 1
# Var ff_c_i_o length 1
# Var ry_i_o length 1
# Var rx_i_o length 1
# Var rc_i_o length 1
# Var nn_c_i_i_o length 1
# Var yy_c_i_i_o length 1
# Var xx_c_i_i_o length 1
# Var ff_c_i_i_o length 1
# Var ry_i_i_o length 1
# Var rx_i_i_o length 1
# Var rc_i_i_o length 1
# Var nn_c_i_i_i_o length 1
# Var yy_c_i_i_i_o length 1
# Var xx_c_i_i_i_o length 1
# Var ff_c_i_i_i_o length 1
# Var ry_i_i_i_o length 1
# Var rx_i_i_i_o length 1
# Var rc_i_i_i_o length 1
# Var nn_c_i_i_i_i length 1
# Var yy_c_i_i_i_i length 1
# Var xx_c_i_i_i_i length 1
# Var ff_c_i_i_i_i length 1
# Var ry_i_i_i_i length 1
# Var rx_i_i_i_i length 1
# Var rc_i_i_i_i length 1
#==--------- Start schedule STAGE filter.shared.wmma.matrix_b ----------==#
s[filter_shared_wmma_matrix_b].compute_at(s[out_wmma_accumulator], rc_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
# Var ax3 length 1
## Tensor core loadB
ax0_o, ax0_i = s[filter_shared_wmma_matrix_b].split(ax0, factor = 1)
ax1_o, ax1_i = s[filter_shared_wmma_matrix_b].split(ax1, factor = 1)
ax2_o, ax2_i = s[filter_shared_wmma_matrix_b].split(ax2, factor = 16)
ax3_o, ax3_i = s[filter_shared_wmma_matrix_b].split(ax3, factor = 16)
s[filter_shared_wmma_matrix_b].reorder(ax0_o, ax1_o, ax2_o, ax3_o, ax0_i, ax1_i, ax2_i, ax3_i, )
s[filter_shared_wmma_matrix_b].tensorize(ax2_i, intrin_wmma_load_matrix_W(
[lb_n0, 1], [sb_n0, 1], (16, 16, 16), row_major, [16, 16], [16, 16], float16, 
))

# Var ax0_o length 1
# Var ax1_o length 1
# Var ax2_o length 1
# Var ax3_o length 1
# Var ax0_i length 1
# Var ax1_i length 1
# Var ax2_i length 1
# Var ax3_i length 1
#==--------- Start schedule STAGE filter.shared ----------==#
s[filter_shared].compute_at(s[out_wmma_accumulator], rc_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
# Var ax3 length 1
## Storage align 
s[filter_shared].storage_align(ax2, 0.000000, 1.000000)
ax0_ax1_f_ax2_f_ax3_f = s[filter_shared].fuse(ax0, ax1, ax2, ax3, )
ax0_ax1_f_ax2_f_ax3_f_o, ax0_ax1_f_ax2_f_ax3_f_i = s[filter_shared].split(ax0_ax1_f_ax2_f_ax3_f, factor = 1)
s[filter_shared].vectorize(ax0_ax1_f_ax2_f_ax3_f_i)
ax0_ax1_f_ax2_f_ax3_f_o_o, ax0_ax1_f_ax2_f_ax3_f_o_i = s[filter_shared].split(ax0_ax1_f_ax2_f_ax3_f_o, factor = 1)
s[filter_shared].bind(ax0_ax1_f_ax2_f_ax3_f_o_i, te.thread_axis("threadIdx.x"))
ax0_ax1_f_ax2_f_ax3_f_o_o_o, ax0_ax1_f_ax2_f_ax3_f_o_o_i = s[filter_shared].split(ax0_ax1_f_ax2_f_ax3_f_o_o, factor = 1)
s[filter_shared].bind(ax0_ax1_f_ax2_f_ax3_f_o_o_i, te.thread_axis("threadIdx.y"))

#==--------- Start schedule STAGE filter ----------==#

#==--------- Start schedule STAGE PaddedInput.shared.wmma.matrix_a ----------==#
s[PaddedInput_shared_wmma_matrix_a].compute_at(s[out_wmma_accumulator], rc_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
# Var ax3 length 1
## Tensor core loadA
ax0_o, ax0_i = s[PaddedInput_shared_wmma_matrix_a].split(ax0, factor = 16)
ax1_o, ax1_i = s[PaddedInput_shared_wmma_matrix_a].split(ax1, factor = 1)
ax2_o, ax2_i = s[PaddedInput_shared_wmma_matrix_a].split(ax2, factor = 1)
ax3_o, ax3_i = s[PaddedInput_shared_wmma_matrix_a].split(ax3, factor = 16)
s[PaddedInput_shared_wmma_matrix_a].reorder(ax0_o, ax1_o, ax2_o, ax3_o, ax0_i, ax1_i, ax2_i, ax3_i, )
s[PaddedInput_shared_wmma_matrix_a].tensorize(ax0_i, intrin_wmma_load_matrix_A(
[la_k0, la_k1, la_k2, 1], [sa_k0, sa_k1, sa_k2, 1], (16, 16, 16), row_major, [16, 1, 1, 16], [16, 1, 1, 16], float16, 
))

# Var ax0_o length 1
# Var ax1_o length 1
# Var ax2_o length 1
# Var ax3_o length 1
# Var ax0_i length 1
# Var ax1_i length 1
# Var ax2_i length 1
# Var ax3_i length 1
#==--------- Start schedule STAGE PaddedInput.shared ----------==#
s[PaddedInput_shared].compute_at(s[out_wmma_accumulator], rc_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
# Var ax3 length 1
## Storage align 
s[PaddedInput_shared].storage_align(ax2, 0.000000, 1.000000)
ax0_ax1_f_ax2_f_ax3_f = s[PaddedInput_shared].fuse(ax0, ax1, ax2, ax3, )
ax0_ax1_f_ax2_f_ax3_f_o, ax0_ax1_f_ax2_f_ax3_f_i = s[PaddedInput_shared].split(ax0_ax1_f_ax2_f_ax3_f, factor = 1)
s[PaddedInput_shared].vectorize(ax0_ax1_f_ax2_f_ax3_f_i)
ax0_ax1_f_ax2_f_ax3_f_o_o, ax0_ax1_f_ax2_f_ax3_f_o_i = s[PaddedInput_shared].split(ax0_ax1_f_ax2_f_ax3_f_o, factor = 1)
s[PaddedInput_shared].bind(ax0_ax1_f_ax2_f_ax3_f_o_i, te.thread_axis("threadIdx.x"))
ax0_ax1_f_ax2_f_ax3_f_o_o_o, ax0_ax1_f_ax2_f_ax3_f_o_o_i = s[PaddedInput_shared].split(ax0_ax1_f_ax2_f_ax3_f_o_o, factor = 1)
s[PaddedInput_shared].bind(ax0_ax1_f_ax2_f_ax3_f_o_o_i, te.thread_axis("threadIdx.y"))

#==--------- Start schedule STAGE input ----------==#
