; void flush_deferred_sse_asm(Deferred *deferred, GList<Vertex> *vertices);

.code

makemask macro a, b, c
    movaps xmm8, a
    xorps  xmm8, b
    psrad  xmm8, 31
    movaps xmmword ptr[rsp + c*16], xmm8
endm

maket macro a, b, c
    movaps xmm8, a
    movaps xmm9, a
    subps  xmm9, b
    divps  xmm8, xmm9
    movaps xmmword ptr[rsp + 12*16 + c*16], xmm8
endm

flush_deferred_sse_asm proc
    sub rsp, 12*16
    movaps xmm0, xmmword ptr[rcx + 0*16]
    movaps xmm1, xmmword ptr[rcx + 1*16]
    movaps xmm2, xmmword ptr[rcx + 2*16]
    movaps xmm3, xmmword ptr[rcx + 3*16]
    movaps xmm4, xmmword ptr[rcx + 4*16]
    movaps xmm5, xmmword ptr[rcx + 5*16]
    movaps xmm6, xmmword ptr[rcx + 6*16]
    movaps xmm7, xmmword ptr[rcx + 7*16]

    makemask xmm0, xmm1, 0
	makemask xmm2, xmm3, 1
	makemask xmm4, xmm5, 2
	makemask xmm6, xmm7, 3
	makemask xmm0, xmm2, 4
	makemask xmm1, xmm3, 5
	makemask xmm4, xmm6, 6
	makemask xmm5, xmm7, 7
	makemask xmm0, xmm4, 8
	makemask xmm1, xmm5, 9
	makemask xmm2, xmm6, 10
	makemask xmm3, xmm7, 11
	; c's are computed
    
	movaps xmm15, xmmword ptr[f1111]
	movaps xmm0,  xmm15
	andps  xmm0,  xmmword ptr[rsp + 0*16]
	movaps xmm1,  xmm15
	andps  xmm1,  xmmword ptr[rsp + 1*16]
	movaps xmm2,  xmm15
	andps  xmm2,  xmmword ptr[rsp + 2*16]
	movaps xmm3,  xmm15
	andps  xmm3,  xmmword ptr[rsp + 3*16]
	movaps xmm4,  xmm15
	andps  xmm4,  xmmword ptr[rsp + 4*16]
	movaps xmm5,  xmm15
	andps  xmm5,  xmmword ptr[rsp + 5*16]
	movaps xmm6,  xmm15
	andps  xmm6,  xmmword ptr[rsp + 6*16]
	movaps xmm7,  xmm15
	andps  xmm7,  xmmword ptr[rsp + 7*16]
	movaps xmm8,  xmm15
	andps  xmm8,  xmmword ptr[rsp + 8*16]
	movaps xmm9,  xmm15
	andps  xmm9,  xmmword ptr[rsp + 9*16]
	movaps xmm10, xmm15
	andps  xmm10, xmmword ptr[rsp + 10*16]
	movaps xmm11, xmm15
	andps  xmm11, xmmword ptr[rsp + 11*16]
	addps xmm0,  xmm1
	addps xmm2,  xmm3
	addps xmm4,  xmm5
	addps xmm6,  xmm7
	addps xmm8,  xmm9
	addps xmm10, xmm11
	addps xmm0, xmm2
	addps xmm4, xmm6
	addps xmm8, xmm10
	addps xmm0, xmm4
	addps xmm0, xmm8



	;maket xmm0, xmm1, 0
	;maket xmm2, xmm3, 1
	;maket xmm4, xmm5, 2
	;maket xmm6, xmm7, 3
	;maket xmm0, xmm2, 4
	;maket xmm1, xmm3, 5
	;maket xmm4, xmm6, 6
	;maket xmm5, xmm7, 7
	;maket xmm0, xmm4, 8
	;maket xmm1, xmm5, 9
	;maket xmm2, xmm6, 10
	;maket xmm3, xmm7, 11

    add rsp, 12*16
    ret
flush_deferred_sse_asm endp

.data
ALIGN 16
f1111 REAL4 1.0, 1.0, 1.0, 1.0

end
