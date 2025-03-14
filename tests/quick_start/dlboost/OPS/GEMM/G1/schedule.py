
## Cache write global
dense_global = s.cache_write(dense, global)

#==--------- Start schedule STAGE dense ----------==#

## Unroll pragma 
i_o, i_i = s[dense].split(i, nparts = 1)
j_o, j_i = s[dense].split(j, nparts = 1)
s[dense].reorder(i_o, j_o, i_i, j_i, )

## Parallel 

## tile spatial 
i_i_o, i_i_i = s[dense].split(i_i, nparts = 1)
j_i_o, j_i_i = s[dense].split(j_i, nparts = 1)
s[dense].reorder(i_i_o, j_i_o, i_i_i, j_i_i, )
i_i_o_j_i_o_f = s[dense].fuse(i_i_o, j_i_o, )
s[dense].parallel(i_i_o_j_i_o_f)

## Tile for cache 

## tile spatial 
i_i_i_o, i_i_i_i = s[dense].split(i_i_i, nparts = 1)
j_i_i_o, j_i_i_i = s[dense].split(j_i_i, nparts = 1)
s[dense].reorder(i_i_i_o, j_i_i_o, i_i_i_i, j_i_i_i, )

# Var i_o length 1
# Var j_o length 1
# Var i_i_o_j_i_o_f length 1
# Var i_i_i_o length 1
# Var j_i_i_o length 1
# Var i_i_i_i length 1
# Var j_i_i_i length 1
#==--------- Start schedule STAGE dense.global ----------==#
s[dense_global].compute_at(s[dense], j_o)

# Var i_c length 1
# Var j_c length 1
# Var k
## general tile 

## tile 
i_c_o, i_c_i = s[dense_global].split(i_c, nparts = 1)
j_c_o, j_c_i = s[dense_global].split(j_c, nparts = 1)
k_o, k_i = s[dense_global].split(k, nparts = 1)
s[dense_global].reorder(i_c_o, j_c_o, k_o, i_c_i, j_c_i, k_i, )

## tile 
i_c_i_o, i_c_i_i = s[dense_global].split(i_c_i, nparts = 1)
j_c_i_o, j_c_i_i = s[dense_global].split(j_c_i, nparts = 1)
k_i_o, k_i_i = s[dense_global].split(k_i, nparts = 1)
s[dense_global].reorder(i_c_i_o, j_c_i_o, k_i_o, i_c_i_i, j_c_i_i, k_i_i, )

## tile 
i_c_i_i_o, i_c_i_i_i = s[dense_global].split(i_c_i_i, nparts = 1)
j_c_i_i_o, j_c_i_i_i = s[dense_global].split(j_c_i_i, nparts = 1)
k_i_i_o, k_i_i_i = s[dense_global].split(k_i_i, nparts = 1)
s[dense_global].reorder(i_c_i_i_o, j_c_i_i_o, k_i_i_o, i_c_i_i_i, j_c_i_i_i, k_i_i_i, )

# Var i_c_o length 1
# Var j_c_o length 1
# Var k_o length 1
# Var i_c_i_o length 1
# Var j_c_i_o length 1
# Var k_i_o length 1
# Var i_c_i_i_o length 1
# Var j_c_i_i_o length 1
# Var k_i_i_o length 1
# Var i_c_i_i_i length 1
# Var j_c_i_i_i length 1
# Var k_i_i_i length 1
#==--------- Start schedule STAGE B ----------==#

#==--------- Start schedule STAGE A ----------==#
