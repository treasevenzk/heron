
## Cache Tensor Core
batch_matmul_wmma_accumulator = s.cache_write(batch_matmul, wmma.accumulator)

## Cache read shared
A_shared = s.cache_read(A, shared, batch_matmul.wmma.accumulator)
B_shared = s.cache_read(B, shared, batch_matmul.wmma.accumulator)
A_shared_wmma_matrix_a = s.cache_read(A_shared, wmma.matrix_a, batch_matmul.wmma.accumulator)
B_shared_wmma_matrix_b = s.cache_read(B_shared, wmma.matrix_b, batch_matmul.wmma.accumulator)

## Cache read shared
batch_matmul_wmma_accumulator_shared = s.cache_read(batch_matmul_wmma_accumulator, shared, batch_matmul)

#==--------- Start schedule STAGE batch_matmul ----------==#

## Unroll pragma 
b_o, b_i = s[batch_matmul].split(b, nparts = 1)
i_o, i_i = s[batch_matmul].split(i, nparts = 1)
j_o, j_i = s[batch_matmul].split(j, nparts = 1)
s[batch_matmul].reorder(b_o, i_o, j_o, b_i, i_i, j_i, )

## Bind blockIdx.x

## tile spatial 
b_i_o, b_i_i = s[batch_matmul].split(b_i, nparts = 1)
i_i_o, i_i_i = s[batch_matmul].split(i_i, nparts = 1)
j_i_o, j_i_i = s[batch_matmul].split(j_i, nparts = 1)
s[batch_matmul].reorder(b_i_o, i_i_o, j_i_o, b_i_i, i_i_i, j_i_i, )
b_i_o_i_i_o_f_j_i_o_f = s[batch_matmul].fuse(b_i_o, i_i_o, j_i_o, )
s[batch_matmul].bind(b_i_o_i_i_o_f_j_i_o_f, te.thread_axis("blockIdx.x"))

## Bind threadIdx.y

## tile spatial 
b_i_i_o, b_i_i_i = s[batch_matmul].split(b_i_i, nparts = 1)
i_i_i_o, i_i_i_i = s[batch_matmul].split(i_i_i, nparts = 1)
j_i_i_o, j_i_i_i = s[batch_matmul].split(j_i_i, nparts = 1)
s[batch_matmul].reorder(b_i_i_o, i_i_i_o, j_i_i_o, b_i_i_i, i_i_i_i, j_i_i_i, )
b_i_i_o_i_i_i_o_f_j_i_i_o_f = s[batch_matmul].fuse(b_i_i_o, i_i_i_o, j_i_i_o, )
s[batch_matmul].bind(b_i_i_o_i_i_i_o_f_j_i_i_o_f, te.thread_axis("threadIdx.y"))

## Bind threadIdx.x

## tile spatial 
b_i_i_i_o, b_i_i_i_i = s[batch_matmul].split(b_i_i_i, nparts = 1)
i_i_i_i_o, i_i_i_i_i = s[batch_matmul].split(i_i_i_i, nparts = 1)
j_i_i_i_o, j_i_i_i_i = s[batch_matmul].split(j_i_i_i, nparts = 1)
s[batch_matmul].reorder(b_i_i_i_o, i_i_i_i_o, j_i_i_i_o, b_i_i_i_i, i_i_i_i_i, j_i_i_i_i, )
b_i_i_i_o_i_i_i_i_o_f_j_i_i_i_o_f = s[batch_matmul].fuse(b_i_i_i_o, i_i_i_i_o, j_i_i_i_o, )
s[batch_matmul].bind(b_i_i_i_o_i_i_i_i_o_f_j_i_i_i_o_f, te.thread_axis("threadIdx.x"))

## Vectorize 

## tile spatial 
b_i_i_i_i_o, b_i_i_i_i_i = s[batch_matmul].split(b_i_i_i_i, nparts = 1)
i_i_i_i_i_o, i_i_i_i_i_i = s[batch_matmul].split(i_i_i_i_i, nparts = 1)
j_i_i_i_i_o, j_i_i_i_i_i = s[batch_matmul].split(j_i_i_i_i, nparts = 1)
s[batch_matmul].reorder(b_i_i_i_i_o, i_i_i_i_i_o, j_i_i_i_i_o, b_i_i_i_i_i, i_i_i_i_i_i, j_i_i_i_i_i, )
b_i_i_i_i_i_i_i_i_i_i_i_f_j_i_i_i_i_i_f = s[batch_matmul].fuse(b_i_i_i_i_i, i_i_i_i_i_i, j_i_i_i_i_i, )
s[batch_matmul].vectorize(b_i_i_i_i_i_i_i_i_i_i_i_f_j_i_i_i_i_i_f)

# Var b_o length 1
# Var i_o length 1
# Var j_o length 1
# Var b_i_o_i_i_o_f_j_i_o_f length 1
# Var b_i_i_o_i_i_i_o_f_j_i_i_o_f length 1
# Var b_i_i_i_o_i_i_i_i_o_f_j_i_i_i_o_f length 1
# Var b_i_i_i_i_o length 1
# Var i_i_i_i_i_o length 1
# Var j_i_i_i_i_o length 1
# Var b_i_i_i_i_i_i_i_i_i_i_i_f_j_i_i_i_i_i_f length 1
#==--------- Start schedule STAGE batch_matmul.wmma.accumulator.shared ----------==#
s[batch_matmul_wmma_accumulator_shared].compute_at(s[batch_matmul], j_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
## Storage align 
s[batch_matmul_wmma_accumulator_shared].storage_align(ax1, 0.000000, 1.000000)

## Bind threadIdx.y

## tile spatial 
ax0_o, ax0_i = s[batch_matmul_wmma_accumulator_shared].split(ax0, nparts = 1)
ax1_o, ax1_i = s[batch_matmul_wmma_accumulator_shared].split(ax1, nparts = 1)
ax2_o, ax2_i = s[batch_matmul_wmma_accumulator_shared].split(ax2, nparts = 1)
s[batch_matmul_wmma_accumulator_shared].reorder(ax0_o, ax1_o, ax2_o, ax0_i, ax1_i, ax2_i, )
ax0_o_ax1_o_f_ax2_o_f = s[batch_matmul_wmma_accumulator_shared].fuse(ax0_o, ax1_o, ax2_o, )
s[batch_matmul_wmma_accumulator_shared].bind(ax0_o_ax1_o_f_ax2_o_f, te.thread_axis("threadIdx.y"))

## Tensor core store
ax0_i_o, ax0_i_i = s[batch_matmul_wmma_accumulator_shared].split(ax0_i, factor = 1)
ax1_i_o, ax1_i_i = s[batch_matmul_wmma_accumulator_shared].split(ax1_i, factor = 16)
ax2_i_o, ax2_i_i = s[batch_matmul_wmma_accumulator_shared].split(ax2_i, factor = 16)
s[batch_matmul_wmma_accumulator_shared].reorder(ax0_i_o, ax1_i_o, ax2_i_o, ax0_i_i, ax1_i_i, ax2_i_i, )
s[batch_matmul_wmma_accumulator_shared].tensorize(ax1_i_i, intrin_wmma_store_matrix(
[sc_n0, 1], [lc_n0, 1], (16, 16, 16), float16, [16, 16], [16, 16], 
))

# Var ax0_o_ax1_o_f_ax2_o_f length 1
# Var ax0_i_o length 1
# Var ax1_i_o length 1
# Var ax2_i_o length 1
# Var ax0_i_i length 1
# Var ax1_i_i length 1
# Var ax2_i_i length 1
#==--------- Start schedule STAGE batch_matmul.wmma.accumulator ----------==#
s[batch_matmul_wmma_accumulator].compute_at(s[batch_matmul_wmma_accumulator_shared], ax0_o_ax1_o_f_ax2_o_f)

# Var b_c length 1
# Var i_c length 1
# Var j_c length 1
# Var k
## general tile 

## tile 
b_c_o, b_c_i = s[batch_matmul_wmma_accumulator].split(b_c, nparts = 1)
i_c_o, i_c_i = s[batch_matmul_wmma_accumulator].split(i_c, nparts = 1)
j_c_o, j_c_i = s[batch_matmul_wmma_accumulator].split(j_c, nparts = 1)
k_o, k_i = s[batch_matmul_wmma_accumulator].split(k, nparts = 1)
s[batch_matmul_wmma_accumulator].reorder(b_c_o, i_c_o, j_c_o, k_o, b_c_i, i_c_i, j_c_i, k_i, )

## tile 
b_c_i_o, b_c_i_i = s[batch_matmul_wmma_accumulator].split(b_c_i, nparts = 1)
i_c_i_o, i_c_i_i = s[batch_matmul_wmma_accumulator].split(i_c_i, nparts = 1)
j_c_i_o, j_c_i_i = s[batch_matmul_wmma_accumulator].split(j_c_i, nparts = 1)
k_i_o, k_i_i = s[batch_matmul_wmma_accumulator].split(k_i, nparts = 1)
s[batch_matmul_wmma_accumulator].reorder(b_c_i_o, i_c_i_o, j_c_i_o, k_i_o, b_c_i_i, i_c_i_i, j_c_i_i, k_i_i, )

## tile 
b_c_i_i_o, b_c_i_i_i = s[batch_matmul_wmma_accumulator].split(b_c_i_i, nparts = 1)
i_c_i_i_o, i_c_i_i_i = s[batch_matmul_wmma_accumulator].split(i_c_i_i, nparts = 1)
j_c_i_i_o, j_c_i_i_i = s[batch_matmul_wmma_accumulator].split(j_c_i_i, nparts = 1)
k_i_i_o, k_i_i_i = s[batch_matmul_wmma_accumulator].split(k_i_i, nparts = 1)
s[batch_matmul_wmma_accumulator].reorder(b_c_i_i_o, i_c_i_i_o, j_c_i_i_o, k_i_i_o, b_c_i_i_i, i_c_i_i_i, j_c_i_i_i, k_i_i_i, )

## Tensor core compute
b_c_i_i_i_o, b_c_i_i_i_i = s[batch_matmul_wmma_accumulator].split(b_c_i_i_i, factor = 1)
i_c_i_i_i_o, i_c_i_i_i_i = s[batch_matmul_wmma_accumulator].split(i_c_i_i_i, factor = 16)
j_c_i_i_i_o, j_c_i_i_i_i = s[batch_matmul_wmma_accumulator].split(j_c_i_i_i, factor = 16)
k_i_i_i_o, k_i_i_i_i = s[batch_matmul_wmma_accumulator].split(k_i_i_i, factor = 16)
s[batch_matmul_wmma_accumulator].reorder(b_c_i_i_i_o, i_c_i_i_i_o, j_c_i_i_i_o, k_i_i_i_o, b_c_i_i_i_i, i_c_i_i_i_i, j_c_i_i_i_i, k_i_i_i_i, )
s[batch_matmul_wmma_accumulator].tensorize(i_c_i_i_i_i, intrin_wmma_gemm(
Tensor(shape=[16, 16], op.name=wmma_A), Tensor(shape=[16, 16], op.name=wmma_B), Tensor(shape=[16, 16], op.name=wmma_C), [la_k0, 1], [lb_k0, 1], [lc_n0, 1], (16, 16, 16), 
))

# Var b_c_o length 1
# Var i_c_o length 1
# Var j_c_o length 1
# Var k_o length 1
# Var b_c_i_o length 1
# Var i_c_i_o length 1
# Var j_c_i_o length 1
# Var k_i_o length 1
# Var b_c_i_i_o length 1
# Var i_c_i_i_o length 1
# Var j_c_i_i_o length 1
# Var k_i_i_o length 1
# Var b_c_i_i_i_o length 1
# Var i_c_i_i_i_o length 1
# Var j_c_i_i_i_o length 1
# Var k_i_i_i_o length 1
# Var b_c_i_i_i_i length 1
# Var i_c_i_i_i_i length 1
# Var j_c_i_i_i_i length 1
# Var k_i_i_i_i length 1
#==--------- Start schedule STAGE B.shared.wmma.matrix_b ----------==#
s[B_shared_wmma_matrix_b].compute_at(s[batch_matmul_wmma_accumulator], k_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
## Tensor core loadB
ax0_o, ax0_i = s[B_shared_wmma_matrix_b].split(ax0, factor = 1)
ax1_o, ax1_i = s[B_shared_wmma_matrix_b].split(ax1, factor = 16)
ax2_o, ax2_i = s[B_shared_wmma_matrix_b].split(ax2, factor = 16)
s[B_shared_wmma_matrix_b].reorder(ax0_o, ax1_o, ax2_o, ax0_i, ax1_i, ax2_i, )
s[B_shared_wmma_matrix_b].tensorize(ax1_i, intrin_wmma_load_matrix_W(
[lb_k0, 1], [sb_k0, 1], (16, 16, 16), col_major, [16, 16], [16, 16], float16, 
))

# Var ax0_o length 1
# Var ax1_o length 1
# Var ax2_o length 1
# Var ax0_i length 1
# Var ax1_i length 1
# Var ax2_i length 1
#==--------- Start schedule STAGE B.shared ----------==#
s[B_shared].compute_at(s[batch_matmul_wmma_accumulator], k_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
## Storage align 
s[B_shared].storage_align(ax1, 0.000000, 1.000000)
ax0_ax1_f_ax2_f = s[B_shared].fuse(ax0, ax1, ax2, )
ax0_ax1_f_ax2_f_o, ax0_ax1_f_ax2_f_i = s[B_shared].split(ax0_ax1_f_ax2_f, factor = 1)
s[B_shared].vectorize(ax0_ax1_f_ax2_f_i)
ax0_ax1_f_ax2_f_o_o, ax0_ax1_f_ax2_f_o_i = s[B_shared].split(ax0_ax1_f_ax2_f_o, factor = 1)
s[B_shared].bind(ax0_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))
ax0_ax1_f_ax2_f_o_o_o, ax0_ax1_f_ax2_f_o_o_i = s[B_shared].split(ax0_ax1_f_ax2_f_o_o, factor = 1)
s[B_shared].bind(ax0_ax1_f_ax2_f_o_o_i, te.thread_axis("threadIdx.y"))

#==--------- Start schedule STAGE B ----------==#

#==--------- Start schedule STAGE A.shared.wmma.matrix_a ----------==#
s[A_shared_wmma_matrix_a].compute_at(s[batch_matmul_wmma_accumulator], k_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
## Tensor core loadA
ax0_o, ax0_i = s[A_shared_wmma_matrix_a].split(ax0, factor = 1)
ax1_o, ax1_i = s[A_shared_wmma_matrix_a].split(ax1, factor = 16)
ax2_o, ax2_i = s[A_shared_wmma_matrix_a].split(ax2, factor = 16)
s[A_shared_wmma_matrix_a].reorder(ax0_o, ax1_o, ax2_o, ax0_i, ax1_i, ax2_i, )
s[A_shared_wmma_matrix_a].tensorize(ax1_i, intrin_wmma_load_matrix_A(
[la_k0, 1], [sa_k0, 1], (16, 16, 16), row_major, [16, 16], [16, 16], float16, 
))

# Var ax0_o length 1
# Var ax1_o length 1
# Var ax2_o length 1
# Var ax0_i length 1
# Var ax1_i length 1
# Var ax2_i length 1
#==--------- Start schedule STAGE A.shared ----------==#
s[A_shared].compute_at(s[batch_matmul_wmma_accumulator], k_o)

# Var ax0 length 1
# Var ax1 length 1
# Var ax2 length 1
## Storage align 
s[A_shared].storage_align(ax1, 0.000000, 1.000000)
ax0_ax1_f_ax2_f = s[A_shared].fuse(ax0, ax1, ax2, )
ax0_ax1_f_ax2_f_o, ax0_ax1_f_ax2_f_i = s[A_shared].split(ax0_ax1_f_ax2_f, factor = 1)
s[A_shared].vectorize(ax0_ax1_f_ax2_f_i)
ax0_ax1_f_ax2_f_o_o, ax0_ax1_f_ax2_f_o_i = s[A_shared].split(ax0_ax1_f_ax2_f_o, factor = 1)
s[A_shared].bind(ax0_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))
ax0_ax1_f_ax2_f_o_o_o, ax0_ax1_f_ax2_f_o_o_i = s[A_shared].split(ax0_ax1_f_ax2_f_o_o, factor = 1)
s[A_shared].bind(ax0_ax1_f_ax2_f_o_o_i, te.thread_axis("threadIdx.y"))

#==--------- Start schedule STAGE A ----------==#
