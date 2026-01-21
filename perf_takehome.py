"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build_vliw(self, slots: list[tuple[str, tuple]]):
        """Pack independent operations into VLIW bundles respecting slot limits."""
        if not slots:
            return []

        bundles = []
        current = defaultdict(list)

        for engine, slot in slots:
            if len(current[engine]) >= SLOT_LIMITS[engine]:
                # Current bundle is full for this engine, emit it
                bundles.append(dict(current))
                current = defaultdict(list)
            current[engine].append(slot)

        if current:
            bundles.append(dict(current))

        return bundles

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle):
        """Add a pre-built instruction bundle."""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vector(self, name=None):
        """Allocate a vector register (VLEN words)."""
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_no_emit(self, val, name=None):
        """Reserve scratch for a constant but don't emit the load instruction yet."""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Heavily optimized VLIW vectorized kernel with deep pipelining.
        Processes 2 vector batches (16 items) per iteration with overlapping.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Scratch space addresses for init vars
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Pre-load all constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pre-load hash constants
        hash_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_consts.append((self.scratch_const(val1), self.scratch_const(val3)))

        self.add("flow", ("pause",))

        # Allocate 3 full sets of vector registers for 3-batch groups
        N_PARALLEL = 3
        idx_vec = [self.alloc_vector(f"idx_vec_{i}") for i in range(N_PARALLEL)]
        val_vec = [self.alloc_vector(f"val_vec_{i}") for i in range(N_PARALLEL)]
        node_vec = [self.alloc_vector(f"node_vec_{i}") for i in range(N_PARALLEL)]
        tmp1_vec = [self.alloc_vector(f"tmp1_vec_{i}") for i in range(N_PARALLEL)]
        tmp2_vec = [self.alloc_vector(f"tmp2_vec_{i}") for i in range(N_PARALLEL)]

        # Allocate scalar addresses for tree node loads (8 addresses per batch)
        node_addrs = [[self.alloc_scratch(f"node_addr_{s}_{i}") for i in range(VLEN)] for s in range(N_PARALLEL)]

        # Broadcast constants to vectors
        zero_vec = self.alloc_vector("zero_vec")
        one_vec = self.alloc_vector("one_vec")
        two_vec = self.alloc_vector("two_vec")
        n_nodes_vec = self.alloc_vector("n_nodes_vec")
        forest_p_vec = self.alloc_vector("forest_p_vec")  # For VALU address computation

        self.add_bundle({"valu": [
            ("vbroadcast", zero_vec, zero_const),
            ("vbroadcast", one_vec, one_const),
            ("vbroadcast", two_vec, two_const),
            ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]),
            ("vbroadcast", forest_p_vec, self.scratch["forest_values_p"]),
        ]})

        # Hash constant vectors
        hash_const_vecs = []
        for i, (const1, const3) in enumerate(hash_consts):
            cv1 = self.alloc_vector(f"hash_c1_{i}")
            cv3 = self.alloc_vector(f"hash_c3_{i}")
            hash_const_vecs.append((cv1, cv3))

        hash_broadcasts = []
        for i, ((const1, const3), (cv1, cv3)) in enumerate(zip(hash_consts, hash_const_vecs)):
            hash_broadcasts.append(("vbroadcast", cv1, const1))
            hash_broadcasts.append(("vbroadcast", cv3, const3))

        for i in range(0, len(hash_broadcasts), 6):
            chunk = hash_broadcasts[i:i+6]
            self.add_bundle({"valu": [("vbroadcast", op[1], op[2]) for op in chunk]})

        # Pre-compute all batch offset constants and MEMORY addresses
        n_batches = batch_size // VLEN
        batch_offset_consts = [self.scratch_const(i * VLEN) for i in range(n_batches)]

        mem_idx_addrs = [self.alloc_scratch(f"mem_idx_addr_{i}") for i in range(n_batches)]
        mem_val_addrs = [self.alloc_scratch(f"mem_val_addr_{i}") for i in range(n_batches)]

        # Compute all memory addresses upfront
        addr_ops = []
        for i in range(n_batches):
            addr_ops.append(("+", mem_idx_addrs[i], self.scratch["inp_indices_p"], batch_offset_consts[i]))
            addr_ops.append(("+", mem_val_addrs[i], self.scratch["inp_values_p"], batch_offset_consts[i]))

        for i in range(0, len(addr_ops), 12):
            chunk = addr_ops[i:i+12]
            self.add_bundle({"alu": chunk})

        # Allocate scratch space for ALL indices and values (to avoid per-round memory access)
        # Each batch is VLEN=8 items, we have n_batches batches
        # Total: n_batches * VLEN * 2 = 256 + 256 = 512 words
        scratch_idx_base = self.alloc_scratch("scratch_idx", n_batches * VLEN)
        scratch_val_base = self.alloc_scratch("scratch_val", n_batches * VLEN)

        # SCRATCH addresses for each batch's idx/val (these are the working locations)
        idx_addrs = [scratch_idx_base + i * VLEN for i in range(n_batches)]
        val_addrs = [scratch_val_base + i * VLEN for i in range(n_batches)]

        # Initial load: copy all idx/val from memory to scratch
        for i in range(n_batches):
            self.add_bundle({"load": [("vload", idx_addrs[i], mem_idx_addrs[i]), ("vload", val_addrs[i], mem_val_addrs[i])]})

        # Note: idx_addrs and val_addrs are now SCRATCH addresses, not pointers to memory
        # We'll need to update the kernel to work with scratch addresses directly

        def emit_debug_compare(vec_base, round_idx, batch_idx, key):
            batch_offset = batch_idx * VLEN
            for vi in range(VLEN):
                self.add_bundle({"debug": [("compare", vec_base + vi, (round_idx, batch_offset + vi, key))]})

        def emit_debug_hash_compare(vec_base, round_idx, batch_idx, stage):
            batch_offset = batch_idx * VLEN
            for vi in range(VLEN):
                self.add_bundle({"debug": [("compare", vec_base + vi, (round_idx, batch_offset + vi, "hash_stage", stage))]})

        # Extra register sets for deep pipelining - Set B (2nd pipeline stage)
        idx_vec_extra = [self.alloc_vector(f"idx_vec_ex_{i}") for i in range(3)]
        val_vec_extra = [self.alloc_vector(f"val_vec_ex_{i}") for i in range(3)]
        node_vec_extra = [self.alloc_vector(f"node_vec_ex_{i}") for i in range(3)]
        tmp1_vec_extra = [self.alloc_vector(f"tmp1_vec_ex_{i}") for i in range(3)]
        tmp2_vec_extra = [self.alloc_vector(f"tmp2_vec_ex_{i}") for i in range(3)]
        node_addrs_extra = [[self.alloc_scratch(f"node_addr_ex_{s}_{i}") for i in range(VLEN)] for s in range(3)]

        # Third register set for 3-deep pipelining - Set C (3rd pipeline stage)
        idx_vec_c = [self.alloc_vector(f"idx_vec_c_{i}") for i in range(3)]
        val_vec_c = [self.alloc_vector(f"val_vec_c_{i}") for i in range(3)]
        node_vec_c = [self.alloc_vector(f"node_vec_c_{i}") for i in range(3)]
        tmp1_vec_c = [self.alloc_vector(f"tmp1_vec_c_{i}") for i in range(3)]
        tmp2_vec_c = [self.alloc_vector(f"tmp2_vec_c_{i}") for i in range(3)]
        node_addrs_c = [[self.alloc_scratch(f"node_addr_c_{s}_{i}") for i in range(VLEN)] for s in range(3)]

        # Group A uses idx_vec[0:3], val_vec[0:3], etc.
        # Group B uses idx_vec_extra[0:3], val_vec_extra[0:3], etc.

        def get_group_regs(use_extra):
            if use_extra:
                return idx_vec_extra, val_vec_extra, node_vec_extra, tmp1_vec_extra, tmp2_vec_extra, node_addrs_extra
            else:
                return idx_vec, val_vec, node_vec, tmp1_vec, tmp2_vec, node_addrs

        def emit_triple_with_overlap(batch_indices, round_idx, prev_ivs=None, prev_vvs=None, prev_t1vs=None, prev_ias=None, prev_vas=None, prev_round=None, prev_batches=None, use_extra=False):
            """Process 3 batches, overlapping tree loads with previous group's hash computation."""
            ivs_list, vvs_list, nvs_list, t1vs_list, t2vs_list, nas_list = get_group_regs(use_extra)
            ivs = ivs_list
            vvs = vvs_list
            nvs = nvs_list
            t1vs = t1vs_list
            t2vs = t2vs_list
            nas = nas_list
            ias = [idx_addrs[bi] for bi in batch_indices]
            vas = [val_addrs[bi] for bi in batch_indices]

            # === Load idx/val for current group ===
            for i in range(3):
                self.add_bundle({"load": [("vload", ivs[i], ias[i]), ("vload", vvs[i], vas[i])]})
                emit_debug_compare(ivs[i], round_idx, batch_indices[i], "idx")
                emit_debug_compare(vvs[i], round_idx, batch_indices[i], "val")

            # Compute node addresses
            all_addr_ops = []
            for i in range(3):
                for vi in range(VLEN):
                    all_addr_ops.append(("+", nas[i][vi], self.scratch["forest_values_p"], ivs[i] + vi))
            for j in range(0, len(all_addr_ops), 12):
                chunk = all_addr_ops[j:j+12]
                self.add_bundle({"alu": chunk})

            # === OVERLAPPED SECTION: Load tree nodes for current + compute hash for previous ===
            if prev_ivs is not None:
                # We have a previous group to compute hash for
                # Interleave: 2 loads + some VALU ops per cycle

                # First, XOR previous group (can do with first tree loads)
                self.add_bundle({
                    "load": [("load", nvs[0], nas[0][0]), ("load", nvs[0] + 1, nas[0][1])],
                    "valu": [("^", prev_vvs[i], prev_vvs[i], nvs_list[i] if not use_extra else node_vec[i]) for i in range(3)]
                })

                # Load remaining tree nodes while doing hash stages
                load_idx = 1  # We've done loads 0,1 of batch 0
                load_ops_remaining = []
                for vi in range(2, VLEN, 2):
                    load_ops_remaining.append((("load", nvs[0] + vi, nas[0][vi]), ("load", nvs[0] + vi + 1, nas[0][vi + 1])))
                for i in range(1, 3):
                    for vi in range(0, VLEN, 2):
                        load_ops_remaining.append((("load", nvs[i] + vi, nas[i][vi]), ("load", nvs[i] + vi + 1, nas[i][vi + 1])))

                # We have 11 more load pairs, and 12 hash cycles to interleave
                hash_cycle = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = []
                    for i in range(3):
                        parallel_ops.append((op1, prev_t1vs[i], prev_vvs[i], cv1))
                        parallel_ops.append((op3, t2vs_list[i] if not use_extra else tmp2_vec[i], prev_vvs[i], cv3))

                    # Parallel ops cycle
                    bundle = {"valu": parallel_ops}
                    if load_idx < len(load_ops_remaining):
                        bundle["load"] = list(load_ops_remaining[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)

                    # Combine ops cycle
                    combine_ops = [(op2, prev_vvs[i], prev_t1vs[i], t2vs_list[i] if not use_extra else tmp2_vec[i]) for i in range(3)]
                    bundle = {"valu": combine_ops}
                    if load_idx < len(load_ops_remaining):
                        bundle["load"] = list(load_ops_remaining[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)

                    for i in range(3):
                        emit_debug_hash_compare(prev_vvs[i], prev_round, prev_batches[i], hi)

                # Finish any remaining loads
                while load_idx < len(load_ops_remaining):
                    self.add_bundle({"load": list(load_ops_remaining[load_idx])})
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(prev_vvs[i], prev_round, prev_batches[i], "hashed_val")

                # Previous group's index computation + store (overlapped with current's remaining work if any)
                self.add_bundle({"valu": [("&", prev_t1vs[i], prev_vvs[i], one_vec) for i in range(3)] +
                                         [("*", prev_ivs[i], prev_ivs[i], two_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", prev_t1vs[i], prev_t1vs[i], one_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", prev_ivs[i], prev_ivs[i], prev_t1vs[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(prev_ivs[i], prev_round, prev_batches[i], "next_idx")

                self.add_bundle({"valu": [("<", prev_t1vs[i], prev_ivs[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", prev_ivs[i], prev_ivs[i], prev_t1vs[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(prev_ivs[i], prev_round, prev_batches[i], "wrapped_idx")

                for i in range(3):
                    self.add_bundle({"store": [("vstore", prev_ias[i], prev_ivs[i]), ("vstore", prev_vas[i], prev_vvs[i])]})

            else:
                # No previous group - just load tree nodes normally
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        self.add_bundle({"load": [("load", nvs[i] + vi, nas[i][vi]), ("load", nvs[i] + vi + 1, nas[i][vi + 1])]})

            for i in range(3):
                emit_debug_compare(nvs[i], round_idx, batch_indices[i], "node_val")

            return ivs, vvs, t1vs, ias, vas

        def emit_final_group_compute(ivs, vvs, nvs, t1vs, t2vs, ias, vas, round_idx, batch_indices, use_extra):
            """Finish computing and storing the final group."""
            # XOR
            if use_extra:
                actual_nvs = node_vec_extra
            else:
                actual_nvs = node_vec
            self.add_bundle({"valu": [("^", vvs[i], vvs[i], actual_nvs[i]) for i in range(3)]})

            # Hash
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                cv1, cv3 = hash_const_vecs[hi]
                parallel_ops = []
                for i in range(3):
                    parallel_ops.append((op1, t1vs[i], vvs[i], cv1))
                    parallel_ops.append((op3, t2vs[i], vvs[i], cv3))
                self.add_bundle({"valu": parallel_ops})
                combine_ops = [(op2, vvs[i], t1vs[i], t2vs[i]) for i in range(3)]
                self.add_bundle({"valu": combine_ops})
                for i in range(3):
                    emit_debug_hash_compare(vvs[i], round_idx, batch_indices[i], hi)

            for i in range(3):
                emit_debug_compare(vvs[i], round_idx, batch_indices[i], "hashed_val")

            # Index compute
            self.add_bundle({"valu": [("&", t1vs[i], vvs[i], one_vec) for i in range(3)] +
                                     [("*", ivs[i], ivs[i], two_vec) for i in range(3)]})
            self.add_bundle({"valu": [("+", t1vs[i], t1vs[i], one_vec) for i in range(3)]})
            self.add_bundle({"valu": [("+", ivs[i], ivs[i], t1vs[i]) for i in range(3)]})
            for i in range(3):
                emit_debug_compare(ivs[i], round_idx, batch_indices[i], "next_idx")

            self.add_bundle({"valu": [("<", t1vs[i], ivs[i], n_nodes_vec) for i in range(3)]})
            self.add_bundle({"valu": [("*", ivs[i], ivs[i], t1vs[i]) for i in range(3)]})
            for i in range(3):
                emit_debug_compare(ivs[i], round_idx, batch_indices[i], "wrapped_idx")

            # Store
            for i in range(3):
                self.add_bundle({"store": [("vstore", ias[i], ivs[i]), ("vstore", vas[i], vvs[i])]})

        def emit_triple_batch_pipelined(batch_indices, round_idx, prev_batch_indices=None, prev_round_idx=None):
            """Fallback: Process 3 batches without overlap."""
            ivs = [idx_vec[i] for i in range(3)]
            vvs = [val_vec[i] for i in range(3)]
            nvs = [node_vec[i] for i in range(3)]
            t1vs = [tmp1_vec[i] for i in range(3)]
            t2vs = [tmp2_vec[i] for i in range(3)]
            nas = [node_addrs[i] for i in range(3)]
            ias = [idx_addrs[bi] for bi in batch_indices]
            vas = [val_addrs[bi] for bi in batch_indices]

            for i in range(3):
                self.add_bundle({"load": [("vload", ivs[i], ias[i]), ("vload", vvs[i], vas[i])]})
                emit_debug_compare(ivs[i], round_idx, batch_indices[i], "idx")
                emit_debug_compare(vvs[i], round_idx, batch_indices[i], "val")

            all_addr_ops = []
            for i in range(3):
                for vi in range(VLEN):
                    all_addr_ops.append(("+", nas[i][vi], self.scratch["forest_values_p"], ivs[i] + vi))
            for j in range(0, len(all_addr_ops), 12):
                self.add_bundle({"alu": all_addr_ops[j:j+12]})

            for vi in range(0, VLEN, 2):
                for i in range(3):
                    self.add_bundle({"load": [("load", nvs[i] + vi, nas[i][vi]), ("load", nvs[i] + vi + 1, nas[i][vi + 1])]})
            for i in range(3):
                emit_debug_compare(nvs[i], round_idx, batch_indices[i], "node_val")

            self.add_bundle({"valu": [("^", vvs[i], vvs[i], nvs[i]) for i in range(3)]})

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                cv1, cv3 = hash_const_vecs[hi]
                parallel_ops = [(op1, t1vs[i], vvs[i], cv1) for i in range(3)] + [(op3, t2vs[i], vvs[i], cv3) for i in range(3)]
                self.add_bundle({"valu": parallel_ops})
                self.add_bundle({"valu": [(op2, vvs[i], t1vs[i], t2vs[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_hash_compare(vvs[i], round_idx, batch_indices[i], hi)

            for i in range(3):
                emit_debug_compare(vvs[i], round_idx, batch_indices[i], "hashed_val")

            self.add_bundle({"valu": [("&", t1vs[i], vvs[i], one_vec) for i in range(3)] +
                                     [("*", ivs[i], ivs[i], two_vec) for i in range(3)]})
            self.add_bundle({"valu": [("+", t1vs[i], t1vs[i], one_vec) for i in range(3)]})
            self.add_bundle({"valu": [("+", ivs[i], ivs[i], t1vs[i]) for i in range(3)]})
            for i in range(3):
                emit_debug_compare(ivs[i], round_idx, batch_indices[i], "next_idx")

            self.add_bundle({"valu": [("<", t1vs[i], ivs[i], n_nodes_vec) for i in range(3)]})
            self.add_bundle({"valu": [("*", ivs[i], ivs[i], t1vs[i]) for i in range(3)]})
            for i in range(3):
                emit_debug_compare(ivs[i], round_idx, batch_indices[i], "wrapped_idx")

            for i in range(3):
                self.add_bundle({"store": [("vstore", ias[i], ivs[i]), ("vstore", vas[i], vvs[i])]})

        def emit_single_batch(batch_idx, round_idx, s):
            """Process a single batch using register set s."""
            iv, vv, nv, t1v, t2v = idx_vec[s], val_vec[s], node_vec[s], tmp1_vec[s], tmp2_vec[s]
            na = node_addrs[s]
            ia, va = idx_addrs[batch_idx], val_addrs[batch_idx]

            self.add_bundle({"load": [("vload", iv, ia), ("vload", vv, va)]})
            emit_debug_compare(iv, round_idx, batch_idx, "idx")
            emit_debug_compare(vv, round_idx, batch_idx, "val")

            self.add_bundle({"alu": [("+", na[vi], self.scratch["forest_values_p"], iv + vi) for vi in range(VLEN)]})

            for vi in range(0, VLEN, 2):
                self.add_bundle({"load": [("load", nv + vi, na[vi]), ("load", nv + vi + 1, na[vi + 1])]})
            emit_debug_compare(nv, round_idx, batch_idx, "node_val")

            self.add_bundle({"valu": [("^", vv, vv, nv)]})
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                cv1, cv3 = hash_const_vecs[hi]
                self.add_bundle({"valu": [(op1, t1v, vv, cv1), (op3, t2v, vv, cv3)]})
                self.add_bundle({"valu": [(op2, vv, t1v, t2v)]})
                emit_debug_hash_compare(vv, round_idx, batch_idx, hi)
            emit_debug_compare(vv, round_idx, batch_idx, "hashed_val")

            self.add_bundle({"valu": [("&", t1v, vv, one_vec), ("*", iv, iv, two_vec)]})
            self.add_bundle({"valu": [("+", t1v, t1v, one_vec)]})
            self.add_bundle({"valu": [("+", iv, iv, t1v)]})
            emit_debug_compare(iv, round_idx, batch_idx, "next_idx")

            self.add_bundle({"valu": [("<", t1v, iv, n_nodes_vec)]})
            self.add_bundle({"valu": [("*", iv, iv, t1v)]})
            emit_debug_compare(iv, round_idx, batch_idx, "wrapped_idx")

            self.add_bundle({"store": [("vstore", ia, iv), ("vstore", va, vv)]})

        def emit_pipelined_round_v2(round_idx):
            """Process all batches using 3-group pipelining with store/load overlap."""
            n_groups = n_batches // 3
            remainder = n_batches % 3

            # We use 2 register sets, alternating
            def get_regs(group_id):
                if group_id % 2 == 0:
                    return idx_vec, val_vec, node_vec, tmp1_vec, tmp2_vec, node_addrs
                else:
                    return idx_vec_extra, val_vec_extra, node_vec_extra, tmp1_vec_extra, tmp2_vec_extra, node_addrs_extra

            # Pipeline state
            loaded_group = -1  # Group whose tree nodes are loaded
            computed_group = -1  # Group whose hash is computed

            # --- STAGE 1: Load group 0 idx/val + tree nodes ---
            if n_groups > 0:
                ivs0, vvs0, nvs0, t1vs0, t2vs0, nas0 = get_regs(0)
                batch0 = [0, 1, 2]
                ias0 = [idx_addrs[bi] for bi in batch0]
                vas0 = [val_addrs[bi] for bi in batch0]

                for i in range(3):
                    self.add_bundle({"load": [("vload", ivs0[i], ias0[i]), ("vload", vvs0[i], vas0[i])]})
                    emit_debug_compare(ivs0[i], round_idx, batch0[i], "idx")
                    emit_debug_compare(vvs0[i], round_idx, batch0[i], "val")

                addr_ops = []
                for i in range(3):
                    for vi in range(VLEN):
                        addr_ops.append(("+", nas0[i][vi], self.scratch["forest_values_p"], ivs0[i] + vi))
                for j in range(0, len(addr_ops), 12):
                    self.add_bundle({"alu": addr_ops[j:j+12]})

                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        self.add_bundle({"load": [("load", nvs0[i] + vi, nas0[i][vi]), ("load", nvs0[i] + vi + 1, nas0[i][vi + 1])]})
                for i in range(3):
                    emit_debug_compare(nvs0[i], round_idx, batch0[i], "node_val")
                loaded_group = 0

            # --- MAIN LOOP ---
            for g in range(1, n_groups):
                ivs_cur, vvs_cur, nvs_cur, t1vs_cur, t2vs_cur, nas_cur = get_regs(g)
                batch_cur = [g*3, g*3+1, g*3+2]
                ias_cur = [idx_addrs[bi] for bi in batch_cur]
                vas_cur = [val_addrs[bi] for bi in batch_cur]

                ivs_prev, vvs_prev, nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ias_prev = [idx_addrs[bi] for bi in batch_prev]
                vas_prev = [val_addrs[bi] for bi in batch_prev]

                # Get store group (g-2) if exists
                has_store = g >= 2
                if has_store:
                    ivs_store, vvs_store, nvs_store, t1vs_store, t2vs_store, nas_store = get_regs(g-2)
                    batch_store = [(g-2)*3, (g-2)*3+1, (g-2)*3+2]
                    ias_store = [idx_addrs[bi] for bi in batch_store]
                    vas_store = [val_addrs[bi] for bi in batch_store]

                # Overlap: load cur idx/val + store g-2 results
                for i in range(3):
                    bundle = {"load": [("vload", ivs_cur[i], ias_cur[i]), ("vload", vvs_cur[i], vas_cur[i])]}
                    if has_store:
                        bundle["store"] = [("vstore", ias_store[i], ivs_store[i]), ("vstore", vas_store[i], vvs_store[i])]
                    self.add_bundle(bundle)
                    emit_debug_compare(ivs_cur[i], round_idx, batch_cur[i], "idx")
                    emit_debug_compare(vvs_cur[i], round_idx, batch_cur[i], "val")

                # Compute cur addresses + XOR prev
                addr_ops = []
                for i in range(3):
                    for vi in range(VLEN):
                        addr_ops.append(("+", nas_cur[i][vi], self.scratch["forest_values_p"], ivs_cur[i] + vi))
                self.add_bundle({
                    "alu": addr_ops[0:12],
                    "valu": [("^", vvs_prev[i], vvs_prev[i], nvs_prev[i]) for i in range(3)]
                })
                if len(addr_ops) > 12:
                    self.add_bundle({"alu": addr_ops[12:24]})

                # Load cur tree nodes + hash prev
                load_idx = 0
                load_sequence = []
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        load_sequence.append((("load", nvs_cur[i] + vi, nas_cur[i][vi]), ("load", nvs_cur[i] + vi + 1, nas_cur[i][vi + 1])))

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = []
                    for i in range(3):
                        parallel_ops.append((op1, t1vs_prev[i], vvs_prev[i], cv1))
                        parallel_ops.append((op3, t2vs_prev[i], vvs_prev[i], cv3))
                    bundle = {"valu": parallel_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)

                    combine_ops = [(op2, vvs_prev[i], t1vs_prev[i], t2vs_prev[i]) for i in range(3)]
                    bundle = {"valu": combine_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_prev[i], round_idx, batch_prev[i], hi)

                while load_idx < len(load_sequence):
                    self.add_bundle({"load": list(load_sequence[load_idx])})
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_prev[i], round_idx, batch_prev[i], "hashed_val")
                for i in range(3):
                    emit_debug_compare(nvs_cur[i], round_idx, batch_cur[i], "node_val")

                # Prev index compute - note: ALU is free during these VALU cycles
                self.add_bundle({"valu": [("&", t1vs_prev[i], vvs_prev[i], one_vec) for i in range(3)] +
                                         [("*", ivs_prev[i], ivs_prev[i], two_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", t1vs_prev[i], t1vs_prev[i], one_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "next_idx")

                # Overlap clamp with looking ahead - but we don't have useful ALU work here
                self.add_bundle({"valu": [("<", t1vs_prev[i], ivs_prev[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "wrapped_idx")

            # --- EPILOGUE: Process remaining groups ---
            if n_groups >= 2:
                # Store group n_groups-2
                ivs_store, vvs_store, _, _, _, _ = get_regs(n_groups - 2)
                batch_store = [(n_groups-2)*3, (n_groups-2)*3+1, (n_groups-2)*3+2]
                ias_store = [idx_addrs[bi] for bi in batch_store]
                vas_store = [val_addrs[bi] for bi in batch_store]
                for i in range(3):
                    self.add_bundle({"store": [("vstore", ias_store[i], ivs_store[i]), ("vstore", vas_store[i], vvs_store[i])]})

            if n_groups >= 1:
                # Compute and store group n_groups-1
                last_g = n_groups - 1
                ivs_last, vvs_last, nvs_last, t1vs_last, t2vs_last, _ = get_regs(last_g)
                batch_last = [last_g*3, last_g*3+1, last_g*3+2]
                ias_last = [idx_addrs[bi] for bi in batch_last]
                vas_last = [val_addrs[bi] for bi in batch_last]

                self.add_bundle({"valu": [("^", vvs_last[i], vvs_last[i], nvs_last[i]) for i in range(3)]})

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs_last[i], vvs_last[i], cv1) for i in range(3)] + [(op3, t2vs_last[i], vvs_last[i], cv3) for i in range(3)]
                    self.add_bundle({"valu": parallel_ops})
                    self.add_bundle({"valu": [(op2, vvs_last[i], t1vs_last[i], t2vs_last[i]) for i in range(3)]})
                    for i in range(3):
                        emit_debug_hash_compare(vvs_last[i], round_idx, batch_last[i], hi)

                for i in range(3):
                    emit_debug_compare(vvs_last[i], round_idx, batch_last[i], "hashed_val")

                self.add_bundle({"valu": [("&", t1vs_last[i], vvs_last[i], one_vec) for i in range(3)] +
                                         [("*", ivs_last[i], ivs_last[i], two_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", t1vs_last[i], t1vs_last[i], one_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "next_idx")

                self.add_bundle({"valu": [("<", t1vs_last[i], ivs_last[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "wrapped_idx")

                for i in range(3):
                    self.add_bundle({"store": [("vstore", ias_last[i], ivs_last[i]), ("vstore", vas_last[i], vvs_last[i])]})

            # Handle remaining batches more efficiently (2 at a time)
            remaining_start = n_groups * 3
            remaining_count = n_batches - remaining_start

            if remaining_count == 2:
                # Process 2 remaining batches together with overlapping
                ivs2 = [idx_vec[0], idx_vec[1]]
                vvs2 = [val_vec[0], val_vec[1]]
                nvs2 = [node_vec[0], node_vec[1]]
                t1vs2 = [tmp1_vec[0], tmp1_vec[1]]
                t2vs2 = [tmp2_vec[0], tmp2_vec[1]]
                nas2 = [node_addrs[0], node_addrs[1]]
                batch2 = [remaining_start, remaining_start + 1]
                ias2 = [idx_addrs[bi] for bi in batch2]
                vas2 = [val_addrs[bi] for bi in batch2]

                # Load idx/val for both
                for i in range(2):
                    self.add_bundle({"load": [("vload", ivs2[i], ias2[i]), ("vload", vvs2[i], vas2[i])]})
                    emit_debug_compare(ivs2[i], round_idx, batch2[i], "idx")
                    emit_debug_compare(vvs2[i], round_idx, batch2[i], "val")

                # Compute addresses (16 ops = 2 cycles)
                addr_ops = []
                for i in range(2):
                    for vi in range(VLEN):
                        addr_ops.append(("+", nas2[i][vi], self.scratch["forest_values_p"], ivs2[i] + vi))
                for j in range(0, len(addr_ops), 12):
                    self.add_bundle({"alu": addr_ops[j:j+12]})

                # Load all tree nodes (16 loads = 8 cycles)
                for vi in range(0, VLEN, 2):
                    for i in range(2):
                        self.add_bundle({"load": [("load", nvs2[i] + vi, nas2[i][vi]), ("load", nvs2[i] + vi + 1, nas2[i][vi + 1])]})

                for i in range(2):
                    emit_debug_compare(nvs2[i], round_idx, batch2[i], "node_val")

                # XOR
                self.add_bundle({"valu": [("^", vvs2[i], vvs2[i], nvs2[i]) for i in range(2)]})

                # Hash stages
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs2[i], vvs2[i], cv1) for i in range(2)] + [(op3, t2vs2[i], vvs2[i], cv3) for i in range(2)]
                    self.add_bundle({"valu": parallel_ops})
                    combine_ops = [(op2, vvs2[i], t1vs2[i], t2vs2[i]) for i in range(2)]
                    self.add_bundle({"valu": combine_ops})
                    for i in range(2):
                        emit_debug_hash_compare(vvs2[i], round_idx, batch2[i], hi)

                for i in range(2):
                    emit_debug_compare(vvs2[i], round_idx, batch2[i], "hashed_val")

                # Index compute for 2 batches (can fit in same cycles)
                self.add_bundle({"valu": [("&", t1vs2[i], vvs2[i], one_vec) for i in range(2)] +
                                         [("*", ivs2[i], ivs2[i], two_vec) for i in range(2)]})
                self.add_bundle({"valu": [("+", t1vs2[i], t1vs2[i], one_vec) for i in range(2)]})
                self.add_bundle({"valu": [("+", ivs2[i], ivs2[i], t1vs2[i]) for i in range(2)]})
                for i in range(2):
                    emit_debug_compare(ivs2[i], round_idx, batch2[i], "next_idx")

                self.add_bundle({"valu": [("<", t1vs2[i], ivs2[i], n_nodes_vec) for i in range(2)]})
                self.add_bundle({"valu": [("*", ivs2[i], ivs2[i], t1vs2[i]) for i in range(2)]})
                for i in range(2):
                    emit_debug_compare(ivs2[i], round_idx, batch2[i], "wrapped_idx")

                # Store both
                for i in range(2):
                    self.add_bundle({"store": [("vstore", ias2[i], ivs2[i]), ("vstore", vas2[i], vvs2[i])]})
            else:
                # Fallback for other remainder counts
                for bi in range(remaining_start, n_batches):
                    emit_single_batch(bi, round_idx, bi % 3)

        def emit_pipelined_scratch_round(round_idx):
            """Process batches using scratch-resident idx/val with tree load / hash pipelining."""
            n_groups = n_batches // 3

            # Use two sets of node vectors and temp vectors for pipelining
            def get_node_regs(group_id):
                if group_id % 2 == 0:
                    return node_vec, tmp1_vec, tmp2_vec, node_addrs
                else:
                    return node_vec_extra, tmp1_vec_extra, tmp2_vec_extra, node_addrs_extra

            # --- PROLOGUE: Load tree nodes for group 0 ---
            if n_groups > 0:
                nvs0, t1vs0, t2vs0, nas0 = get_node_regs(0)
                batch0 = [0, 1, 2]
                ivs0 = [idx_addrs[b] for b in batch0]
                vvs0 = [val_addrs[b] for b in batch0]

                for i in range(3):
                    emit_debug_compare(ivs0[i], round_idx, batch0[i], "idx")
                    emit_debug_compare(vvs0[i], round_idx, batch0[i], "val")

                # Compute tree node addresses using VALU (3 ops instead of 24 ALU ops)
                self.add_bundle({"valu": [("+", nas0[i][0], forest_p_vec, ivs0[i]) for i in range(3)]})

                # Load tree nodes for group 0
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        self.add_bundle({"load": [("load", nvs0[i] + vi, nas0[i][vi]), ("load", nvs0[i] + vi + 1, nas0[i][vi + 1])]})
                for i in range(3):
                    emit_debug_compare(nvs0[i], round_idx, batch0[i], "node_val")

            # --- MAIN LOOP with 3-deep pipelining: Load[G] + Hash[G-1] + Index[G-2] ---
            # First iteration (g=1): Load[1] + Hash[0], no Index yet
            if n_groups > 1:
                g = 1
                nvs_cur, t1vs_cur, t2vs_cur, nas_cur = get_node_regs(g)
                batch_cur = [g*3, g*3+1, g*3+2]
                ivs_cur = [idx_addrs[b] for b in batch_cur]
                vvs_cur = [val_addrs[b] for b in batch_cur]

                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                for i in range(3):
                    emit_debug_compare(ivs_cur[i], round_idx, batch_cur[i], "idx")
                    emit_debug_compare(vvs_cur[i], round_idx, batch_cur[i], "val")

                # Use VALU for address computation (3 ops) + XOR (3 ops) = 6 ops in 1 cycle
                self.add_bundle({
                    "valu": [("+", nas_cur[i][0], forest_p_vec, ivs_cur[i]) for i in range(3)] +
                            [("^", vvs_prev[i], vvs_prev[i], nvs_prev[i]) for i in range(3)]
                })

                load_sequence = []
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        load_sequence.append((("load", nvs_cur[i] + vi, nas_cur[i][vi]), ("load", nvs_cur[i] + vi + 1, nas_cur[i][vi + 1])))

                load_idx = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs_prev[i], vvs_prev[i], cv1) for i in range(3)] + [(op3, t2vs_prev[i], vvs_prev[i], cv3) for i in range(3)]
                    bundle = {"valu": parallel_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)
                    combine_ops = [(op2, vvs_prev[i], t1vs_prev[i], t2vs_prev[i]) for i in range(3)]
                    bundle = {"valu": combine_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_prev[i], round_idx, batch_prev[i], hi)

                while load_idx < len(load_sequence):
                    self.add_bundle({"load": list(load_sequence[load_idx])})
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_prev[i], round_idx, batch_prev[i], "hashed_val")
                for i in range(3):
                    emit_debug_compare(nvs_cur[i], round_idx, batch_cur[i], "node_val")

            # Main iterations (g=2 to n_groups-1): Load[G] + Hash[G-1] + Index[G-2]
            for g in range(2, n_groups):
                nvs_cur, t1vs_cur, t2vs_cur, nas_cur = get_node_regs(g)
                batch_cur = [g*3, g*3+1, g*3+2]
                ivs_cur = [idx_addrs[b] for b in batch_cur]
                vvs_cur = [val_addrs[b] for b in batch_cur]

                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                nvs_pp, t1vs_pp, t2vs_pp, nas_pp = get_node_regs(g-2)
                batch_pp = [(g-2)*3, (g-2)*3+1, (g-2)*3+2]
                ivs_pp = [idx_addrs[b] for b in batch_pp]
                vvs_pp = [val_addrs[b] for b in batch_pp]

                for i in range(3):
                    emit_debug_compare(ivs_cur[i], round_idx, batch_cur[i], "idx")
                    emit_debug_compare(vvs_cur[i], round_idx, batch_cur[i], "val")

                # Addr[G] + XOR[G-1] + Index part 1 [G-2]
                addr_ops = []
                for i in range(3):
                    for vi in range(VLEN):
                        addr_ops.append(("+", nas_cur[i][vi], self.scratch["forest_values_p"], ivs_cur[i] + vi))

                self.add_bundle({
                    "alu": addr_ops[0:12],
                    "valu": [("^", vvs_prev[i], vvs_prev[i], nvs_prev[i]) for i in range(3)] +
                            [("&", t1vs_pp[i], vvs_pp[i], one_vec) for i in range(3)]
                })
                if len(addr_ops) > 12:
                    self.add_bundle({
                        "alu": addr_ops[12:24],
                        "valu": [("*", ivs_pp[i], ivs_pp[i], two_vec) for i in range(3)]
                    })
                else:
                    self.add_bundle({"valu": [("*", ivs_pp[i], ivs_pp[i], two_vec) for i in range(3)]})

                load_sequence = []
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        load_sequence.append((("load", nvs_cur[i] + vi, nas_cur[i][vi]), ("load", nvs_cur[i] + vi + 1, nas_cur[i][vi + 1])))

                # Hash[G-1] + Load[G] + Index[G-2] interleaved
                # Index remaining ops: + (t1+1), + (iv+t1), < (iv < n_nodes), * (iv * mask)
                load_idx = 0
                idx_step = 0  # Track index computation progress
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]

                    # Parallel hash ops + index ops
                    valu_ops = [(op1, t1vs_prev[i], vvs_prev[i], cv1) for i in range(3)] + \
                               [(op3, t2vs_prev[i], vvs_prev[i], cv3) for i in range(3)]

                    bundle = {"valu": valu_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)

                    # Combine + index ops
                    valu_ops = [(op2, vvs_prev[i], t1vs_prev[i], t2vs_prev[i]) for i in range(3)]
                    if idx_step == 0:
                        valu_ops += [("+", t1vs_pp[i], t1vs_pp[i], one_vec) for i in range(3)]
                        idx_step = 1
                    elif idx_step == 1:
                        valu_ops += [("+", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]
                        idx_step = 2
                    elif idx_step == 2:
                        for i in range(3):
                            emit_debug_compare(ivs_pp[i], round_idx, batch_pp[i], "next_idx")
                        valu_ops += [("<", t1vs_pp[i], ivs_pp[i], n_nodes_vec) for i in range(3)]
                        idx_step = 3
                    elif idx_step == 3:
                        valu_ops += [("*", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]
                        idx_step = 4

                    bundle = {"valu": valu_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_prev[i], round_idx, batch_prev[i], hi)

                while load_idx < len(load_sequence):
                    self.add_bundle({"load": list(load_sequence[load_idx])})
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_prev[i], round_idx, batch_prev[i], "hashed_val")
                for i in range(3):
                    emit_debug_compare(nvs_cur[i], round_idx, batch_cur[i], "node_val")

                # Finish any remaining index ops for G-2
                if idx_step < 4:
                    if idx_step <= 1:
                        self.add_bundle({"valu": [("+", t1vs_pp[i], t1vs_pp[i], one_vec) for i in range(3)]})
                    if idx_step <= 2:
                        self.add_bundle({"valu": [("+", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]})
                        for i in range(3):
                            emit_debug_compare(ivs_pp[i], round_idx, batch_pp[i], "next_idx")
                    if idx_step <= 3:
                        self.add_bundle({"valu": [("<", t1vs_pp[i], ivs_pp[i], n_nodes_vec) for i in range(3)]})
                    self.add_bundle({"valu": [("*", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_pp[i], round_idx, batch_pp[i], "wrapped_idx")

            # Post-loop: finish index for groups n_groups-2 and n_groups-1
            if n_groups > 1:
                # Index for group n_groups-2 (was G-1 in last iteration)
                g = n_groups - 1
                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                self.add_bundle({"valu": [("&", t1vs_prev[i], vvs_prev[i], one_vec) for i in range(3)] +
                                         [("*", ivs_prev[i], ivs_prev[i], two_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", t1vs_prev[i], t1vs_prev[i], one_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "next_idx")
                self.add_bundle({"valu": [("<", t1vs_prev[i], ivs_prev[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "wrapped_idx")

            elif n_groups == 1:
                # Only one group, need to do index for group 0
                g = 0
                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g)
                batch_prev = [0, 1, 2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                self.add_bundle({"valu": [("&", t1vs_prev[i], vvs_prev[i], one_vec) for i in range(3)] +
                                         [("*", ivs_prev[i], ivs_prev[i], two_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", t1vs_prev[i], t1vs_prev[i], one_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "next_idx")
                self.add_bundle({"valu": [("<", t1vs_prev[i], ivs_prev[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "wrapped_idx")

            # --- EPILOGUE + REMAINING: Overlap epilogue compute with remaining tree loads ---
            remaining_start = n_groups * 3
            remaining_count = n_batches - remaining_start

            if n_groups >= 1 and remaining_count == 2:
                # Overlap: compute last group while loading tree nodes for remaining 2 batches
                last_g = n_groups - 1
                nvs_last, t1vs_last, t2vs_last, nas_last = get_node_regs(last_g)
                batch_last = [last_g*3, last_g*3+1, last_g*3+2]
                ivs_last = [idx_addrs[b] for b in batch_last]
                vvs_last = [val_addrs[b] for b in batch_last]

                # Remaining batches setup
                bi0, bi1 = remaining_start, remaining_start + 1
                iv0, iv1 = idx_addrs[bi0], idx_addrs[bi1]
                vv0, vv1 = val_addrs[bi0], val_addrs[bi1]
                nv0, nv1 = node_vec[0], node_vec[1]
                t1v0, t1v1 = tmp1_vec[0], tmp1_vec[1]
                t2v0, t2v1 = tmp2_vec[0], tmp2_vec[1]
                na0, na1 = node_addrs[0], node_addrs[1]

                emit_debug_compare(iv0, round_idx, bi0, "idx")
                emit_debug_compare(vv0, round_idx, bi0, "val")
                emit_debug_compare(iv1, round_idx, bi1, "idx")
                emit_debug_compare(vv1, round_idx, bi1, "val")

                # Compute addresses for remaining batches (16 ALU ops) + XOR for epilogue
                addr_ops = []
                for vi in range(VLEN):
                    addr_ops.append(("+", na0[vi], self.scratch["forest_values_p"], iv0 + vi))
                    addr_ops.append(("+", na1[vi], self.scratch["forest_values_p"], iv1 + vi))

                self.add_bundle({
                    "valu": [("^", vvs_last[i], vvs_last[i], nvs_last[i]) for i in range(3)],
                    "alu": addr_ops[0:12]
                })
                self.add_bundle({"alu": addr_ops[12:16]})

                # Prepare load sequence for remaining batches (16 loads)
                rem_load_sequence = []
                for vi in range(0, VLEN, 2):
                    rem_load_sequence.append([("load", nv0 + vi, na0[vi]), ("load", nv0 + vi + 1, na0[vi + 1])])
                    rem_load_sequence.append([("load", nv1 + vi, na1[vi]), ("load", nv1 + vi + 1, na1[vi + 1])])

                # Hash epilogue while loading remaining tree nodes
                load_idx = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs_last[i], vvs_last[i], cv1) for i in range(3)] + [(op3, t2vs_last[i], vvs_last[i], cv3) for i in range(3)]
                    bundle = {"valu": parallel_ops}
                    if load_idx < len(rem_load_sequence):
                        bundle["load"] = rem_load_sequence[load_idx]
                        load_idx += 1
                    self.add_bundle(bundle)

                    combine_ops = [(op2, vvs_last[i], t1vs_last[i], t2vs_last[i]) for i in range(3)]
                    bundle = {"valu": combine_ops}
                    if load_idx < len(rem_load_sequence):
                        bundle["load"] = rem_load_sequence[load_idx]
                        load_idx += 1
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_last[i], round_idx, batch_last[i], hi)

                # Finish any remaining loads
                while load_idx < len(rem_load_sequence):
                    self.add_bundle({"load": rem_load_sequence[load_idx]})
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_last[i], round_idx, batch_last[i], "hashed_val")

                emit_debug_compare(nv0, round_idx, bi0, "node_val")
                emit_debug_compare(nv1, round_idx, bi1, "node_val")

                # Epilogue index compute (can overlap with remaining XOR+hash start)
                # XOR remaining
                self.add_bundle({
                    "valu": [("&", t1vs_last[i], vvs_last[i], one_vec) for i in range(3)] +
                            [("*", ivs_last[i], ivs_last[i], two_vec) for i in range(3)]
                })

                self.add_bundle({
                    "valu": [("+", t1vs_last[i], t1vs_last[i], one_vec) for i in range(3)] +
                            [("^", vv0, vv0, nv0), ("^", vv1, vv1, nv1)]
                })

                self.add_bundle({"valu": [("+", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "next_idx")

                self.add_bundle({"valu": [("<", t1vs_last[i], ivs_last[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "wrapped_idx")

                # Remaining batches: hash + index (epilogue is done)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    self.add_bundle({"valu": [(op1, t1v0, vv0, cv1), (op3, t2v0, vv0, cv3),
                                              (op1, t1v1, vv1, cv1), (op3, t2v1, vv1, cv3)]})
                    self.add_bundle({"valu": [(op2, vv0, t1v0, t2v0), (op2, vv1, t1v1, t2v1)]})
                    emit_debug_hash_compare(vv0, round_idx, bi0, hi)
                    emit_debug_hash_compare(vv1, round_idx, bi1, hi)

                emit_debug_compare(vv0, round_idx, bi0, "hashed_val")
                emit_debug_compare(vv1, round_idx, bi1, "hashed_val")

                self.add_bundle({"valu": [("&", t1v0, vv0, one_vec), ("*", iv0, iv0, two_vec),
                                          ("&", t1v1, vv1, one_vec), ("*", iv1, iv1, two_vec)]})
                self.add_bundle({"valu": [("+", t1v0, t1v0, one_vec), ("+", t1v1, t1v1, one_vec)]})
                self.add_bundle({"valu": [("+", iv0, iv0, t1v0), ("+", iv1, iv1, t1v1)]})
                emit_debug_compare(iv0, round_idx, bi0, "next_idx")
                emit_debug_compare(iv1, round_idx, bi1, "next_idx")

                self.add_bundle({"valu": [("<", t1v0, iv0, n_nodes_vec), ("<", t1v1, iv1, n_nodes_vec)]})
                self.add_bundle({"valu": [("*", iv0, iv0, t1v0), ("*", iv1, iv1, t1v1)]})
                emit_debug_compare(iv0, round_idx, bi0, "wrapped_idx")
                emit_debug_compare(iv1, round_idx, bi1, "wrapped_idx")

            elif n_groups >= 1:
                # No remaining batches, just compute epilogue
                last_g = n_groups - 1
                nvs_last, t1vs_last, t2vs_last, nas_last = get_node_regs(last_g)
                batch_last = [last_g*3, last_g*3+1, last_g*3+2]
                ivs_last = [idx_addrs[b] for b in batch_last]
                vvs_last = [val_addrs[b] for b in batch_last]

                self.add_bundle({"valu": [("^", vvs_last[i], vvs_last[i], nvs_last[i]) for i in range(3)]})

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs_last[i], vvs_last[i], cv1) for i in range(3)] + [(op3, t2vs_last[i], vvs_last[i], cv3) for i in range(3)]
                    self.add_bundle({"valu": parallel_ops})
                    self.add_bundle({"valu": [(op2, vvs_last[i], t1vs_last[i], t2vs_last[i]) for i in range(3)]})
                    for i in range(3):
                        emit_debug_hash_compare(vvs_last[i], round_idx, batch_last[i], hi)

                for i in range(3):
                    emit_debug_compare(vvs_last[i], round_idx, batch_last[i], "hashed_val")

                self.add_bundle({"valu": [("&", t1vs_last[i], vvs_last[i], one_vec) for i in range(3)] +
                                         [("*", ivs_last[i], ivs_last[i], two_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", t1vs_last[i], t1vs_last[i], one_vec) for i in range(3)]})
                self.add_bundle({"valu": [("+", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "next_idx")

                self.add_bundle({"valu": [("<", t1vs_last[i], ivs_last[i], n_nodes_vec) for i in range(3)]})
                self.add_bundle({"valu": [("*", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]})
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "wrapped_idx")

                # Handle remaining batches separately if any (but not 2)
                if remaining_count == 1:
                    bi = remaining_start
                    iv = idx_addrs[bi]
                    vv = val_addrs[bi]
                    nv = node_vec[0]
                    t1v = tmp1_vec[0]
                    t2v = tmp2_vec[0]
                    na = node_addrs[0]

                    emit_debug_compare(iv, round_idx, bi, "idx")
                    emit_debug_compare(vv, round_idx, bi, "val")

                    self.add_bundle({"alu": [("+", na[vi], self.scratch["forest_values_p"], iv + vi) for vi in range(VLEN)]})

                    for vi in range(0, VLEN, 2):
                        self.add_bundle({"load": [("load", nv + vi, na[vi]), ("load", nv + vi + 1, na[vi + 1])]})
                    emit_debug_compare(nv, round_idx, bi, "node_val")

                    self.add_bundle({"valu": [("^", vv, vv, nv)]})
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        cv1, cv3 = hash_const_vecs[hi]
                        self.add_bundle({"valu": [(op1, t1v, vv, cv1), (op3, t2v, vv, cv3)]})
                        self.add_bundle({"valu": [(op2, vv, t1v, t2v)]})
                        emit_debug_hash_compare(vv, round_idx, bi, hi)
                    emit_debug_compare(vv, round_idx, bi, "hashed_val")

                    self.add_bundle({"valu": [("&", t1v, vv, one_vec), ("*", iv, iv, two_vec)]})
                    self.add_bundle({"valu": [("+", t1v, t1v, one_vec)]})
                    self.add_bundle({"valu": [("+", iv, iv, t1v)]})
                    emit_debug_compare(iv, round_idx, bi, "next_idx")

                    self.add_bundle({"valu": [("<", t1v, iv, n_nodes_vec)]})
                    self.add_bundle({"valu": [("*", iv, iv, t1v)]})
                    emit_debug_compare(iv, round_idx, bi, "wrapped_idx")
            elif remaining_count == 1:
                # Single remaining batch
                bi = remaining_start
                iv = idx_addrs[bi]
                vv = val_addrs[bi]
                nv = node_vec[0]
                t1v = tmp1_vec[0]
                t2v = tmp2_vec[0]
                na = node_addrs[0]

                emit_debug_compare(iv, round_idx, bi, "idx")
                emit_debug_compare(vv, round_idx, bi, "val")

                self.add_bundle({"alu": [("+", na[vi], self.scratch["forest_values_p"], iv + vi) for vi in range(VLEN)]})

                for vi in range(0, VLEN, 2):
                    self.add_bundle({"load": [("load", nv + vi, na[vi]), ("load", nv + vi + 1, na[vi + 1])]})
                emit_debug_compare(nv, round_idx, bi, "node_val")

                self.add_bundle({"valu": [("^", vv, vv, nv)]})
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    self.add_bundle({"valu": [(op1, t1v, vv, cv1), (op3, t2v, vv, cv3)]})
                    self.add_bundle({"valu": [(op2, vv, t1v, t2v)]})
                    emit_debug_hash_compare(vv, round_idx, bi, hi)
                emit_debug_compare(vv, round_idx, bi, "hashed_val")

                self.add_bundle({"valu": [("&", t1v, vv, one_vec), ("*", iv, iv, two_vec)]})
                self.add_bundle({"valu": [("+", t1v, t1v, one_vec)]})
                self.add_bundle({"valu": [("+", iv, iv, t1v)]})
                emit_debug_compare(iv, round_idx, bi, "next_idx")

                self.add_bundle({"valu": [("<", t1v, iv, n_nodes_vec)]})
                self.add_bundle({"valu": [("*", iv, iv, t1v)]})
                emit_debug_compare(iv, round_idx, bi, "wrapped_idx")

        # Process all rounds, overlapping final stores with last round's epilogue
        def emit_final_round_with_stores():
            """Special handling for last round - overlap final stores with computation."""
            round_idx = rounds - 1
            n_groups = n_batches // 3

            def get_node_regs(group_id):
                if group_id % 2 == 0:
                    return node_vec, tmp1_vec, tmp2_vec, node_addrs
                else:
                    return node_vec_extra, tmp1_vec_extra, tmp2_vec_extra, node_addrs_extra

            # Track which batches are complete and ready to store
            store_queue = []  # List of batch indices ready to store
            store_ptr = [0]   # Pointer to next batch to store

            def maybe_add_store(bundle):
                """Add a vstore operation if we have batches ready to store."""
                if store_ptr[0] < len(store_queue):
                    bi = store_queue[store_ptr[0]]
                    bundle["store"] = [("vstore", mem_idx_addrs[bi], idx_addrs[bi]),
                                       ("vstore", mem_val_addrs[bi], val_addrs[bi])]
                    store_ptr[0] += 1

            # Prologue - load tree nodes for group 0
            if n_groups > 0:
                nvs0, t1vs0, t2vs0, nas0 = get_node_regs(0)
                batch0 = [0, 1, 2]
                ivs0 = [idx_addrs[b] for b in batch0]
                vvs0 = [val_addrs[b] for b in batch0]

                for i in range(3):
                    emit_debug_compare(ivs0[i], round_idx, batch0[i], "idx")
                    emit_debug_compare(vvs0[i], round_idx, batch0[i], "val")

                # Use VALU for address computation (3 ops instead of 24 ALU ops)
                self.add_bundle({"valu": [("+", nas0[i][0], forest_p_vec, ivs0[i]) for i in range(3)]})

                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        self.add_bundle({"load": [("load", nvs0[i] + vi, nas0[i][vi]), ("load", nvs0[i] + vi + 1, nas0[i][vi + 1])]})
                for i in range(3):
                    emit_debug_compare(nvs0[i], round_idx, batch0[i], "node_val")

            # First iteration (g=1): Load[1] + Hash[0]
            if n_groups > 1:
                g = 1
                nvs_cur, t1vs_cur, t2vs_cur, nas_cur = get_node_regs(g)
                batch_cur = [g*3, g*3+1, g*3+2]
                ivs_cur = [idx_addrs[b] for b in batch_cur]
                vvs_cur = [val_addrs[b] for b in batch_cur]

                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                for i in range(3):
                    emit_debug_compare(ivs_cur[i], round_idx, batch_cur[i], "idx")
                    emit_debug_compare(vvs_cur[i], round_idx, batch_cur[i], "val")

                # Use VALU for address computation (3 ops) + XOR (3 ops) = 6 ops in 1 cycle
                self.add_bundle({
                    "valu": [("+", nas_cur[i][0], forest_p_vec, ivs_cur[i]) for i in range(3)] +
                            [("^", vvs_prev[i], vvs_prev[i], nvs_prev[i]) for i in range(3)]
                })

                load_sequence = []
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        load_sequence.append((("load", nvs_cur[i] + vi, nas_cur[i][vi]), ("load", nvs_cur[i] + vi + 1, nas_cur[i][vi + 1])))

                load_idx = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs_prev[i], vvs_prev[i], cv1) for i in range(3)] + [(op3, t2vs_prev[i], vvs_prev[i], cv3) for i in range(3)]
                    bundle = {"valu": parallel_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)
                    combine_ops = [(op2, vvs_prev[i], t1vs_prev[i], t2vs_prev[i]) for i in range(3)]
                    bundle = {"valu": combine_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_prev[i], round_idx, batch_prev[i], hi)

                while load_idx < len(load_sequence):
                    self.add_bundle({"load": list(load_sequence[load_idx])})
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_prev[i], round_idx, batch_prev[i], "hashed_val")
                for i in range(3):
                    emit_debug_compare(nvs_cur[i], round_idx, batch_cur[i], "node_val")

            # Main loop with 3-deep pipelining + final stores overlap
            for g in range(2, n_groups):
                nvs_cur, t1vs_cur, t2vs_cur, nas_cur = get_node_regs(g)
                batch_cur = [g*3, g*3+1, g*3+2]
                ivs_cur = [idx_addrs[b] for b in batch_cur]
                vvs_cur = [val_addrs[b] for b in batch_cur]

                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                nvs_pp, t1vs_pp, t2vs_pp, nas_pp = get_node_regs(g-2)
                batch_pp = [(g-2)*3, (g-2)*3+1, (g-2)*3+2]
                ivs_pp = [idx_addrs[b] for b in batch_pp]
                vvs_pp = [val_addrs[b] for b in batch_pp]

                for i in range(3):
                    emit_debug_compare(ivs_cur[i], round_idx, batch_cur[i], "idx")
                    emit_debug_compare(vvs_cur[i], round_idx, batch_cur[i], "val")

                addr_ops = []
                for i in range(3):
                    for vi in range(VLEN):
                        addr_ops.append(("+", nas_cur[i][vi], self.scratch["forest_values_p"], ivs_cur[i] + vi))

                bundle = {
                    "alu": addr_ops[0:12],
                    "valu": [("^", vvs_prev[i], vvs_prev[i], nvs_prev[i]) for i in range(3)] +
                            [("&", t1vs_pp[i], vvs_pp[i], one_vec) for i in range(3)]
                }
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                if len(addr_ops) > 12:
                    bundle = {"alu": addr_ops[12:24], "valu": [("*", ivs_pp[i], ivs_pp[i], two_vec) for i in range(3)]}
                else:
                    bundle = {"valu": [("*", ivs_pp[i], ivs_pp[i], two_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                load_sequence = []
                for vi in range(0, VLEN, 2):
                    for i in range(3):
                        load_sequence.append((("load", nvs_cur[i] + vi, nas_cur[i][vi]), ("load", nvs_cur[i] + vi + 1, nas_cur[i][vi + 1])))

                load_idx = 0
                idx_step = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    valu_ops = [(op1, t1vs_prev[i], vvs_prev[i], cv1) for i in range(3)] + \
                               [(op3, t2vs_prev[i], vvs_prev[i], cv3) for i in range(3)]
                    bundle = {"valu": valu_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)

                    valu_ops = [(op2, vvs_prev[i], t1vs_prev[i], t2vs_prev[i]) for i in range(3)]
                    if idx_step == 0:
                        valu_ops += [("+", t1vs_pp[i], t1vs_pp[i], one_vec) for i in range(3)]
                        idx_step = 1
                    elif idx_step == 1:
                        valu_ops += [("+", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]
                        idx_step = 2
                    elif idx_step == 2:
                        for i in range(3):
                            emit_debug_compare(ivs_pp[i], round_idx, batch_pp[i], "next_idx")
                        valu_ops += [("<", t1vs_pp[i], ivs_pp[i], n_nodes_vec) for i in range(3)]
                        idx_step = 3
                    elif idx_step == 3:
                        valu_ops += [("*", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]
                        idx_step = 4

                    bundle = {"valu": valu_ops}
                    if load_idx < len(load_sequence):
                        bundle["load"] = list(load_sequence[load_idx])
                        load_idx += 1
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_prev[i], round_idx, batch_prev[i], hi)

                while load_idx < len(load_sequence):
                    bundle = {"load": list(load_sequence[load_idx])}
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_prev[i], round_idx, batch_prev[i], "hashed_val")
                for i in range(3):
                    emit_debug_compare(nvs_cur[i], round_idx, batch_cur[i], "node_val")

                if idx_step < 4:
                    if idx_step <= 1:
                        bundle = {"valu": [("+", t1vs_pp[i], t1vs_pp[i], one_vec) for i in range(3)]}
                        maybe_add_store(bundle)
                        self.add_bundle(bundle)
                    if idx_step <= 2:
                        bundle = {"valu": [("+", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]}
                        maybe_add_store(bundle)
                        self.add_bundle(bundle)
                        for i in range(3):
                            emit_debug_compare(ivs_pp[i], round_idx, batch_pp[i], "next_idx")
                    if idx_step <= 3:
                        bundle = {"valu": [("<", t1vs_pp[i], ivs_pp[i], n_nodes_vec) for i in range(3)]}
                        maybe_add_store(bundle)
                        self.add_bundle(bundle)
                    bundle = {"valu": [("*", ivs_pp[i], ivs_pp[i], t1vs_pp[i]) for i in range(3)]}
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)
                for i in range(3):
                    emit_debug_compare(ivs_pp[i], round_idx, batch_pp[i], "wrapped_idx")

                # Mark group g-2 as complete and ready to store
                for b in batch_pp:
                    store_queue.append(b)

            # Post-loop: finish index for group n_groups-2
            if n_groups > 1:
                g = n_groups - 1
                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g-1)
                batch_prev = [(g-1)*3, (g-1)*3+1, (g-1)*3+2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                bundle = {"valu": [("&", t1vs_prev[i], vvs_prev[i], one_vec) for i in range(3)] +
                                  [("*", ivs_prev[i], ivs_prev[i], two_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", t1vs_prev[i], t1vs_prev[i], one_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "next_idx")

                bundle = {"valu": [("<", t1vs_prev[i], ivs_prev[i], n_nodes_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("*", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "wrapped_idx")

                # Mark group n_groups-2 as complete
                for b in batch_prev:
                    store_queue.append(b)

            elif n_groups == 1:
                g = 0
                nvs_prev, t1vs_prev, t2vs_prev, nas_prev = get_node_regs(g)
                batch_prev = [0, 1, 2]
                ivs_prev = [idx_addrs[b] for b in batch_prev]
                vvs_prev = [val_addrs[b] for b in batch_prev]

                bundle = {"valu": [("&", t1vs_prev[i], vvs_prev[i], one_vec) for i in range(3)] +
                                  [("*", ivs_prev[i], ivs_prev[i], two_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", t1vs_prev[i], t1vs_prev[i], one_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "next_idx")

                bundle = {"valu": [("<", t1vs_prev[i], ivs_prev[i], n_nodes_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("*", ivs_prev[i], ivs_prev[i], t1vs_prev[i]) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                for i in range(3):
                    emit_debug_compare(ivs_prev[i], round_idx, batch_prev[i], "wrapped_idx")

            # Epilogue + remaining with stores overlap
            remaining_start = n_groups * 3
            remaining_count = n_batches - remaining_start

            if n_groups >= 1 and remaining_count == 2:
                last_g = n_groups - 1
                nvs_last, t1vs_last, t2vs_last, nas_last = get_node_regs(last_g)
                batch_last = [last_g*3, last_g*3+1, last_g*3+2]
                ivs_last = [idx_addrs[b] for b in batch_last]
                vvs_last = [val_addrs[b] for b in batch_last]

                bi0, bi1 = remaining_start, remaining_start + 1
                iv0, iv1 = idx_addrs[bi0], idx_addrs[bi1]
                vv0, vv1 = val_addrs[bi0], val_addrs[bi1]
                nv0, nv1 = node_vec[0], node_vec[1]
                t1v0, t1v1 = tmp1_vec[0], tmp1_vec[1]
                t2v0, t2v1 = tmp2_vec[0], tmp2_vec[1]
                na0, na1 = node_addrs[0], node_addrs[1]

                emit_debug_compare(iv0, round_idx, bi0, "idx")
                emit_debug_compare(vv0, round_idx, bi0, "val")
                emit_debug_compare(iv1, round_idx, bi1, "idx")
                emit_debug_compare(vv1, round_idx, bi1, "val")

                addr_ops = []
                for vi in range(VLEN):
                    addr_ops.append(("+", na0[vi], self.scratch["forest_values_p"], iv0 + vi))
                    addr_ops.append(("+", na1[vi], self.scratch["forest_values_p"], iv1 + vi))

                bundle = {
                    "valu": [("^", vvs_last[i], vvs_last[i], nvs_last[i]) for i in range(3)],
                    "alu": addr_ops[0:12]
                }
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"alu": addr_ops[12:16]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                rem_load_sequence = []
                for vi in range(0, VLEN, 2):
                    rem_load_sequence.append([("load", nv0 + vi, na0[vi]), ("load", nv0 + vi + 1, na0[vi + 1])])
                    rem_load_sequence.append([("load", nv1 + vi, na1[vi]), ("load", nv1 + vi + 1, na1[vi + 1])])

                load_idx = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    parallel_ops = [(op1, t1vs_last[i], vvs_last[i], cv1) for i in range(3)] + [(op3, t2vs_last[i], vvs_last[i], cv3) for i in range(3)]
                    bundle = {"valu": parallel_ops}
                    if load_idx < len(rem_load_sequence):
                        bundle["load"] = rem_load_sequence[load_idx]
                        load_idx += 1
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)

                    combine_ops = [(op2, vvs_last[i], t1vs_last[i], t2vs_last[i]) for i in range(3)]
                    bundle = {"valu": combine_ops}
                    if load_idx < len(rem_load_sequence):
                        bundle["load"] = rem_load_sequence[load_idx]
                        load_idx += 1
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)
                    for i in range(3):
                        emit_debug_hash_compare(vvs_last[i], round_idx, batch_last[i], hi)

                while load_idx < len(rem_load_sequence):
                    bundle = {"load": rem_load_sequence[load_idx]}
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)
                    load_idx += 1

                for i in range(3):
                    emit_debug_compare(vvs_last[i], round_idx, batch_last[i], "hashed_val")
                emit_debug_compare(nv0, round_idx, bi0, "node_val")
                emit_debug_compare(nv1, round_idx, bi1, "node_val")

                # Index for last group + XOR remaining
                bundle = {"valu": [("&", t1vs_last[i], vvs_last[i], one_vec) for i in range(3)] +
                                  [("*", ivs_last[i], ivs_last[i], two_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", t1vs_last[i], t1vs_last[i], one_vec) for i in range(3)] +
                                  [("^", vv0, vv0, nv0), ("^", vv1, vv1, nv1)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "next_idx")

                bundle = {"valu": [("<", t1vs_last[i], ivs_last[i], n_nodes_vec) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("*", ivs_last[i], ivs_last[i], t1vs_last[i]) for i in range(3)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)
                for i in range(3):
                    emit_debug_compare(ivs_last[i], round_idx, batch_last[i], "wrapped_idx")

                # Mark last group as complete
                for b in batch_last:
                    store_queue.append(b)

                # Hash + index for remaining batches
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    cv1, cv3 = hash_const_vecs[hi]
                    bundle = {"valu": [(op1, t1v0, vv0, cv1), (op3, t2v0, vv0, cv3),
                                       (op1, t1v1, vv1, cv1), (op3, t2v1, vv1, cv3)]}
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)

                    bundle = {"valu": [(op2, vv0, t1v0, t2v0), (op2, vv1, t1v1, t2v1)]}
                    maybe_add_store(bundle)
                    self.add_bundle(bundle)
                    emit_debug_hash_compare(vv0, round_idx, bi0, hi)
                    emit_debug_hash_compare(vv1, round_idx, bi1, hi)

                emit_debug_compare(vv0, round_idx, bi0, "hashed_val")
                emit_debug_compare(vv1, round_idx, bi1, "hashed_val")

                bundle = {"valu": [("&", t1v0, vv0, one_vec), ("*", iv0, iv0, two_vec),
                                   ("&", t1v1, vv1, one_vec), ("*", iv1, iv1, two_vec)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", t1v0, t1v0, one_vec), ("+", t1v1, t1v1, one_vec)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("+", iv0, iv0, t1v0), ("+", iv1, iv1, t1v1)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)
                emit_debug_compare(iv0, round_idx, bi0, "next_idx")
                emit_debug_compare(iv1, round_idx, bi1, "next_idx")

                bundle = {"valu": [("<", t1v0, iv0, n_nodes_vec), ("<", t1v1, iv1, n_nodes_vec)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)

                bundle = {"valu": [("*", iv0, iv0, t1v0), ("*", iv1, iv1, t1v1)]}
                maybe_add_store(bundle)
                self.add_bundle(bundle)
                emit_debug_compare(iv0, round_idx, bi0, "wrapped_idx")
                emit_debug_compare(iv1, round_idx, bi1, "wrapped_idx")

                # Mark remaining as complete
                store_queue.append(bi0)
                store_queue.append(bi1)

            # Store any remaining batches
            while store_ptr[0] < len(store_queue):
                bi = store_queue[store_ptr[0]]
                self.add_bundle({"store": [("vstore", mem_idx_addrs[bi], idx_addrs[bi]),
                                           ("vstore", mem_val_addrs[bi], val_addrs[bi])]})
                store_ptr[0] += 1

        # Process rounds 0 to rounds-2 normally
        for round_idx in range(rounds - 1):
            emit_pipelined_scratch_round(round_idx)

        # Process last round with overlapped final stores
        emit_final_round_with_stores()

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
