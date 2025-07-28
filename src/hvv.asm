.code
bench_h proc
    mov ecx, 256
_loop:
    test ecx, ecx
    jz _ret
    vdpps xmm0, xmm0, xmm0, 255
    vdpps xmm1, xmm1, xmm1, 255
    vdpps xmm2, xmm2, xmm2, 255
    vdpps xmm3, xmm3, xmm3, 255
    dec ecx
    jmp _loop
_ret:
    ret
bench_h endp
bench_v proc
    mov ecx, 256
_loop:
    test ecx, ecx
    jz _ret
    vmulps xmm0, xmm0, xmm0
    vmulps xmm1, xmm1, xmm1
    vmulps xmm2, xmm2, xmm2
    vmulps xmm3, xmm3, xmm3
    vaddps xmm0, xmm0, xmm1
    vaddps xmm2, xmm2, xmm3
    vaddps xmm0, xmm0, xmm2
    dec ecx
    jmp _loop
_ret:
    ret
bench_v endp
end