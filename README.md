# Optimized SDF -> Triangle list

![Screenshot](./README/screenshot.png)

* SSE4.1 intrinsics
* Auxiliary bitfields for fast skipping of empty or solid areas

## Tests

Version|Speed
-|-
Initial|~400 mb/s
Optimized|~3 gb/s
