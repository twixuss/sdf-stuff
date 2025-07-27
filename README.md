# Optimized SDF -> Triangle list

![Screenshot](./README/screenshot.png)

* 128^3^
* SSE4.1 intrinsics
* Auxiliary bitfields for fast skipping of empty or solid areas

# Tests

Version|Speed
-|-
Initial|~400 mb/s
Optimized|~3 gb/s

# Ways of producing Normals
## Generate only positions, calculate normals from triangles
<img title="Normals generated only from positions of the triangles" src="./README/normals_triangle.png" width=50% />

This is a "standard" way of calculating normals for a triangle mesh.
After generating positions and triangles, you loop over the triangles, add the normal of the triangle to the vertices' normals.
Then in another loop you normalize the normals, averaging them.
It does look good, but requires processing after generating triangles.
## Calculate from sdf differences when generating a vertex
<img title="Normals generated per vertex based on SDF" src="./README/normals_vertex.png" width=50%/>

When generating a vertex, you can look at differences of distances on each axis and calculate the normal vector from that.
They are calculated significantly faster than from triangles (1.5x) - you do just a little extra work per vertex and don't have to loop over them again.
But it looks significantly worse. Normal is calculated as if it was at the center of the cell, but the position is shifted to make the mesh smoother.
That creates areas where normals are stretched or shrinked way too much, making it look blocky and noisy.

## Calculate from sdf differences in the shader
<img title="Normals generated per pixel based on 3d SDF texture" src="./README/normals_3dtexture.png" width=50% />

The third approach is the most expensive one, the most accurate and it looks the best.
It requires sending _the entire 3d array_ of distances to the GPU, doing _27 fetches_ from that array _per pixel_, then doing a ton of additions to calculate normals of 8 cells, then blending between them.

I don't know if I calculated this correctly, but GPU performance is 2.5x worse than in both previous methods.

.|.
-|-
per vertex normals | 3400 fps at 40% usage
per pixel normals | 2650 fps at 80% usage