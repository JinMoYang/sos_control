package com.example.activeperception.acquire

import kotlin.math.log2
import kotlin.math.max
import kotlin.math.pow

/**
 * Nelder-Mead exposure control (Shin et al., IROS 2019, Algorithm 1) as an
 * inverted-control state machine. Port of sense/nelder_mead.py (class NM), which in turn
 * mirrors the authors' nelder_mead_AE.m. The Python file is the golden source:
 * [NelderMeadTest] replays traces generated from it by sense/nm_golden_gen.py.
 *
 * Each objective evaluation costs one CAPTURE (one frame), so control is inverted into
 * two calls with the invariant "at most ONE point is awaiting evaluation at any time":
 *
 *     val u = nm.propose()   // stop-space (isoStops, shutterStops) to capture next;
 *                            // null once converged. Idempotent (does not change state).
 *     nm.observe(v)          // value of the captured point; advances exactly one step.
 *                            // NM MINIMISES v: pass -f(I) to maximise a quality score f.
 *
 * Search space: u = (log2(ISO/100), log2(480*shutter_s)) clipped to [0, uMax]^2.
 * uMax = 4.0 reproduces the 5x5 simulation; the app's 3x3 grid passes 2.0. Proposals are
 * continuous; the caller snaps to its capture grid with [snapToGrid] (the authors'
 * FindClosestPoint).
 *
 * The intensity-based initial simplex (EPS = 1.7, Algorithm 1) is built in LINEAR
 * (iso, shutter_s) units around x0 = (100*2^u0[0], (1/480)*2^u0[1]), exactly as the
 * Python does. The anchor constants 100 and 1/480 only shape the simplex — they set
 * where the multiplicative perturbation x0*(1+h) lands after the log2 map back to
 * stops — the stop-space output is what the caller consumes; the grid's real
 * ISO/shutter values never enter here.
 *
 * Faithfulness notes carried over from the Python: rho=1, xi=2, gam=0.5, sig=0.5,
 * tol=1e-5, max_feval=50 and the branch inequalities follow nelder_mead_AE.m; shrink
 * does not re-evaluate the best vertex; bounds clip to [0, uMax]; minimisation is
 * realised by feeding -f.
 */
class NelderMead(u0: DoubleArray, meanI: Double, private val uMax: Double) {

    companion object {
        const val RHO = 1.0
        const val XI = 2.0
        const val GAM = 0.5
        const val SIG = 0.5
        const val TOL = 1e-5
        const val MAXF = 50
        const val EPS = 1.7

        /**
         * Round each stop to the nearest integer cell index, clipped; cell =
         * isoIdx * nShutter + shIdx. Mirrors to5/FindClosestPoint. Math.rint is
         * half-to-even, matching Python round() in the golden source (round(2.5) = 2).
         */
        fun snapToGrid(u: DoubleArray, nIso: Int, nShutter: Int): Int {
            val i = Math.rint(u[0]).toInt().coerceIn(0, nIso - 1)
            val s = Math.rint(u[1]).toInt().coerceIn(0, nShutter - 1)
            return i * nShutter + s
        }
    }

    private enum class Kind { NONE, REFLECT, EXPAND, OUTSIDE, INSIDE, SHRINK }

    private var X: Array<DoubleArray>
    private val F = DoubleArray(3)          // valid once the matching seed is observed
    private val pending = ArrayDeque<Int>() // vertex indices awaiting their seed capture
    private var stageKind = Kind.NONE
    private var stagePoint = DoubleArray(2)
    private var xbar = DoubleArray(2)
    private var fR = 0.0                    // reflect value, kept for expand/outside tests
    private var xR = DoubleArray(2)
    private var shrinkI = 0

    var converged = false; private set
    var nFeval = 0; private set

    init {
        // intensity-based initial simplex, vertices x_i = x0*(1+h e_i) in LINEAR units
        val j = meanI
        val h = if (j >= 128) -(j / 255.0) / EPS else EPS * (1 - j / 255.0)
        val x0 = doubleArrayOf(100.0 * 2.0.pow(u0[0]), 1.0 / (480.0 / 2.0.pow(u0[1])))
        X = arrayOf(
            toU(x0[0], x0[1]),
            toU(x0[0] * (1 + h), x0[1]),
            toU(x0[0], x0[1] * (1 + h)),
        )
        pending.addAll(listOf(0, 1, 2))
    }

    /** linear (iso, shutter_s) -> stops, clipped to the grid hull */
    private fun toU(iso: Double, shutterS: Double) = doubleArrayOf(
        log2(max(iso, 1e-6) / 100.0).coerceIn(0.0, uMax),
        log2(max(shutterS, 1e-9) * 480.0).coerceIn(0.0, uMax),
    )

    /** Return the point u to capture next (null if converged). */
    fun propose(): DoubleArray? {
        if (converged) return null
        if (pending.isNotEmpty()) return X[pending.first()].copyOf()
        return stagePoint.copyOf()
    }

    /** Feed -f (we minimise) of the captured point; advance the NM state machine. */
    fun observe(fval: Double) {
        nFeval++
        if (pending.isNotEmpty()) {
            F[pending.removeFirst()] = fval
            if (pending.isEmpty()) { sort(); startIter() }
            return
        }
        val x = stagePoint
        val f1 = F[0]; val f2 = F[1]; val f3 = F[2]
        when (stageKind) {
            Kind.REFLECT -> {
                fR = fval; xR = x
                when {
                    f1 <= fval && fval <= f2 -> accept(x, fval)
                    fval < f1 -> setStage(Kind.EXPAND, clip(comb(1 + RHO * XI, xbar, -(RHO * XI), X[2])))
                    f2 <= fval && fval < f3 -> setStage(Kind.OUTSIDE, clip(comb(1 + RHO * GAM, xbar, -(RHO * GAM), X[2])))
                    else -> setStage(Kind.INSIDE, clip(comb(1 - GAM, xbar, GAM, X[2])))
                }
            }
            Kind.EXPAND -> if (fval < fR) accept(x, fval) else accept(xR, fR)
            Kind.OUTSIDE -> if (fval <= fR) accept(x, fval) else shrink()
            Kind.INSIDE -> if (fval < f3) accept(x, fval) else shrink()
            Kind.SHRINK -> {
                // shrink points stay inside the hull (convex combo of clipped vertices): no clip,
                // matching the Python
                X[shrinkI] = x; F[shrinkI] = fval
                shrinkI++
                if (shrinkI <= 2) setStage(Kind.SHRINK, comb(SIG, X[shrinkI], 1 - SIG, X[0]))
                else { sort(); check(); startIter() }
            }
            Kind.NONE -> error("observe() without a pending proposal")
        }
    }

    fun best(): DoubleArray = X[0].copyOf()

    private fun setStage(kind: Kind, point: DoubleArray) {
        stageKind = kind; stagePoint = point
    }

    private fun accept(x: DoubleArray, f: Double) {
        X[2] = x; F[2] = f
        sort(); check(); startIter()
    }

    private fun shrink() {
        shrinkI = 1
        setStage(Kind.SHRINK, comb(SIG, X[1], 1 - SIG, X[0]))
    }

    private fun sort() {
        // np.argsort over 3 values; ties don't occur in practice (distinct captures)
        val o = listOf(0, 1, 2).sortedBy { F[it] }
        val xs = arrayOf(X[o[0]], X[o[1]], X[o[2]])
        val fs = doubleArrayOf(F[o[0]], F[o[1]], F[o[2]])
        X = xs
        for (i in 0..2) F[i] = fs[i]
    }

    private fun check() {
        if (F[2] - F[0] < TOL || nFeval > MAXF) converged = true
    }

    private fun startIter() {
        if (converged) return
        xbar = doubleArrayOf((X[0][0] + X[1][0]) / 2, (X[0][1] + X[1][1]) / 2)
        setStage(Kind.REFLECT, clip(comb(1 + RHO, xbar, -RHO, X[2])))
    }

    private fun clip(u: DoubleArray) =
        doubleArrayOf(u[0].coerceIn(0.0, uMax), u[1].coerceIn(0.0, uMax))

    /** a*u + b*v, elementwise; b carries the sign so op order matches the numpy expressions. */
    private fun comb(a: Double, u: DoubleArray, b: Double, v: DoubleArray) =
        doubleArrayOf(a * u[0] + b * v[0], a * u[1] + b * v[1])
}
