//! Architecture-specific Swift ABI adapters for Foundation Models.

use core::arch::global_asm;

// Swift reserves x8 for indirect results, x20 for the synchronous context, and
// x22 for the async context on AArch64.
#[cfg(target_arch = "aarch64")]
global_asm!(
	r#"
.text

.p2align 2
.globl _apple_fm_value_get
_apple_fm_value_get:
	mov x8, x0
	br x1

.p2align 2
.globl _apple_fm_availability_get
_apple_fm_availability_get:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	mov x8, x0
	mov x20, x1
	blr x2
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret

.p2align 2
.globl _apple_fm_model_init
_apple_fm_model_init:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	mov x20, x2
	blr x3
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret

.p2align 2
.globl _apple_fm_session_init
_apple_fm_session_init:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	mov x20, x3
	mov x3, x2
	mov x2, x1
	mov x1, x4
	blr x5
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret

.p2align 2
.globl _apple_fm_options_init
_apple_fm_options_init:
	mov x8, x0
	mov x0, x1
	mov x1, x2
	mov x2, x3
	mov x3, x4
	mov x4, x5
	br x6

.p2align 2
.globl _apple_fm_stream_response
_apple_fm_stream_response:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	mov x20, x1
	mov x8, x0
	mov x0, x2
	mov x1, x3
	mov x2, x4
	blr x5
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret

.p2align 2
.globl _apple_fm_make_iterator
_apple_fm_make_iterator:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	mov x8, x0
	mov x0, x2
	mov x20, x1
	blr x3
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret

.p2align 2
.globl _apple_fm_snapshot_content
_apple_fm_snapshot_content:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	mov x20, x0
	mov x8, x2
	mov x0, x1
	blr x3
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret

.p2align 2
.globl _apple_fm_task_create
_apple_fm_task_create:
	mov x6, x3
	mov x4, x2
	mov x2, x1
	adrp x3, _apple_fm_task_entry@PAGE
	add x3, x3, _apple_fm_task_entry@PAGEOFF
	mov x1, #0
	mov w5, #48
	br x6

.p2align 2
_apple_fm_task_entry:
	orr x29, x29, #0x1000000000000000
	sub sp, sp, #32
	stp x29, x30, [sp, #16]
	str x22, [sp, #8]
	add x29, sp, #16
	ldp x8, x9, [x20, #32]
	ldr w0, [x8, #4]
	blr x9
	mov x8, x0
	adrp x9, _apple_fm_next_continuation@PAGE
	add x9, x9, _apple_fm_next_continuation@PAGEOFF
	stp x22, x9, [x0]
	str x20, [x22, #32]
	ldp x0, x4, [x20, #16]
	ldp x20, x3, [x20]
	mov x1, #0
	mov x2, #0
	mov x22, x8
	ldp x29, x30, [sp, #16]
	and x29, x29, #0xefffffffffffffff
	add sp, sp, #32
	br x4

.p2align 2
_apple_fm_next_continuation:
	orr x29, x29, #0x1000000000000000
	str x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	str x22, [sp, #8]
	add x29, sp, #16
	mov x0, x22
	ldr x22, [x22]
	ldr x19, [x22, #32]
	ldr x8, [x19, #48]
	blr x8
	mov x0, x19
	mov x1, x20
	bl _apple_fm_next_completed
	ldr x0, [x22, #8]
	ldp x29, x30, [sp, #16]
	ldr x19, [sp], #32
	and x29, x29, #0xefffffffffffffff
	br x0
"#
);

// Swift reserves rax for indirect results, r13 for the synchronous context,
// and r14 for the async context on AMD64.
#[cfg(target_arch = "x86_64")]
global_asm!(
	r#"
.text

.p2align 4, 0x90
.globl _apple_fm_value_get
_apple_fm_value_get:
	pushq %rbp
	movq %rsp, %rbp
	movq %rdi, %rax
	popq %rbp
	jmpq *%rsi

.p2align 4, 0x90
.globl _apple_fm_availability_get
_apple_fm_availability_get:
	pushq %rbp
	movq %rsp, %rbp
	pushq %r13
	pushq %rax
	movq %rdi, %rax
	movq %rsi, %r13
	callq *%rdx
	addq $8, %rsp
	popq %r13
	popq %rbp
	retq

.p2align 4, 0x90
.globl _apple_fm_model_init
_apple_fm_model_init:
	pushq %rbp
	movq %rsp, %rbp
	pushq %r13
	pushq %rax
	movq %rdx, %r13
	callq *%rcx
	addq $8, %rsp
	popq %r13
	popq %rbp
	retq

.p2align 4, 0x90
.globl _apple_fm_session_init
_apple_fm_session_init:
	pushq %rbp
	movq %rsp, %rbp
	pushq %r13
	pushq %rax
	movq %rcx, %r13
	movq %rdx, %rcx
	movq %rsi, %rdx
	movq %r8, %rsi
	callq *%r9
	addq $8, %rsp
	popq %r13
	popq %rbp
	retq

.p2align 4, 0x90
.globl _apple_fm_options_init
_apple_fm_options_init:
	pushq %rbp
	movq %rsp, %rbp
	movq %r8, %r10
	movq %rdi, %rax
	movq %rsi, %rdi
	movq %rdx, %rsi
	movl %ecx, %edx
	movl %r9d, %r8d
	movq %r10, %rcx
	popq %rbp
	jmpq *8(%rsp)

.p2align 4, 0x90
.globl _apple_fm_stream_response
_apple_fm_stream_response:
	pushq %rbp
	movq %rsp, %rbp
	pushq %r13
	pushq %rax
	movq %rsi, %r13
	movq %rdi, %rax
	movq %rdx, %rdi
	movq %rcx, %rsi
	movq %r8, %rdx
	callq *%r9
	addq $8, %rsp
	popq %r13
	popq %rbp
	retq

.p2align 4, 0x90
.globl _apple_fm_make_iterator
_apple_fm_make_iterator:
	pushq %rbp
	movq %rsp, %rbp
	pushq %r13
	pushq %rax
	movq %rdi, %rax
	movq %rdx, %rdi
	movq %rsi, %r13
	callq *%rcx
	addq $8, %rsp
	popq %r13
	popq %rbp
	retq

.p2align 4, 0x90
.globl _apple_fm_snapshot_content
_apple_fm_snapshot_content:
	pushq %rbp
	movq %rsp, %rbp
	pushq %r13
	pushq %rax
	movq %rdx, %rax
	movq %rdi, %r13
	movq %rsi, %rdi
	callq *%rcx
	addq $8, %rsp
	popq %r13
	popq %rbp
	retq

.p2align 4, 0x90
.globl _apple_fm_task_create
_apple_fm_task_create:
	pushq %rbp
	movq %rsp, %rbp
	movq %rcx, %rax
	movq %rsi, %r10
	leaq _apple_fm_task_entry(%rip), %rcx
	movl $48, %r9d
	movq %rdx, %r8
	xorl %esi, %esi
	movq %r10, %rdx
	popq %rbp
	jmpq *%rax

.p2align 4, 0x90
_apple_fm_task_entry:
	btsq $60, %rbp
	pushq %rbp
	pushq %r14
	leaq 8(%rsp), %rbp
	subq $24, %rsp
	movq 32(%r13), %rax
	movl 4(%rax), %edi
	callq *40(%r13)
	movq %r14, (%rax)
	leaq _apple_fm_next_continuation(%rip), %rcx
	movq %rcx, 8(%rax)
	movq %r13, 32(%r14)
	movq 24(%r13), %r9
	movq 16(%r13), %rdi
	movq (%r13), %r8
	movq 8(%r13), %rcx
	xorl %esi, %esi
	xorl %edx, %edx
	movq %rax, %r14
	movq %r8, %r13
	addq $16, %rsp
	addq $16, %rsp
	popq %rbp
	btrq $60, %rbp
	jmpq *%r9

.p2align 4, 0x90
_apple_fm_next_continuation:
	btsq $60, %rbp
	pushq %rbp
	pushq %r14
	leaq 8(%rsp), %rbp
	subq $8, %rsp
	pushq %rbx
	subq $24, %rsp
	movq %r14, %rdi
	movq (%r14), %r14
	movq 32(%r14), %rbx
	callq *48(%rbx)
	movq %rbx, %rdi
	movq %r13, %rsi
	callq _apple_fm_next_completed
	movq 8(%r14), %rax
	addq $24, %rsp
	popq %rbx
	addq $16, %rsp
	popq %rbp
	btrq $60, %rbp
	jmpq *%rax
"#,
	options(att_syntax)
);
