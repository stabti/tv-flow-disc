# How to make a disc disappear with maths

Total variation (TV) flow applied to two discs. Nothing shrinks: the discs lose contrast at a precise rate until they vanish, the small one first.

[Watch the video](media/tv_disc.mp4)

## What you are seeing

The TV flow evolves an image by smoothing its contours, i.e. by reducing their length:

    du/dt = div( grad(u) / |grad(u)| )

For a disc of radius R and contrast h on a black background, the solution stays an exact disc:

    u(t) = max(h - 2t/R, 0) on the disc, 0 elsewhere

- The contrast decreases linearly at rate 2/R.
- The disc disappears at the finite time t* = hR/2.
- Half the radius means twice the speed: here R=15 vanishes at t=7.5 and R=30 at t=15.

In other words the disc is a nonlinear eigenfunction of the TV operator. This is why TV denoising removes small contrasted structures first, and why it biases the contrast of the structures it keeps.

## Background

This reproduces the disc example analysed by Yves Meyer in *Oscillating Patterns in Image Processing and Nonlinear Evolution Equations* (AMS, 2001), where it illustrates how the Rudin-Osher-Fatemi model behaves as a function of scale. The same behaviour appears in the TV flow, which is what is simulated here.

For me this is also a chance to revisit topics I studied during my postdoc in 2017-2018.

## Implementation

- TV flow = implicit Euler scheme: one ROF denoising problem per time step (lambda = 1/dt).
- Each ROF problem is solved in its dual form (Chambolle-style projection, FISTA for the 16-neighbour scheme), warm-started from the previous step.
- The image is zero outside the frame (Dirichlet), which matches the theory on the whole plane. With Neumann borders the mass of the disc would spread over the background.
- Two discretisations of TV (`--scheme`):
  - `crofton` (default): 16 neighbours weighted with Cauchy-Crofton weights. The perimeter of a digital disc is within 1% of 2*pi*R and discs stay round.
  - `iso`: forward differences. Faster, but it favours axis-aligned edges, so discs turn slightly polygonal.

The simulated contrast at the disc centres follows the theoretical line h - 2t/R.

## Run

Requires Python 3, numpy, matplotlib and ffmpeg.

    pip install numpy matplotlib
    python tv_flow_disc.py --out tv_disc.mp4                # with plot and annotations
    python tv_flow_disc.py --clean --out tv_disc_clean.mp4  # discs only

`--lang fr` gives French labels. Runtime is about 40 seconds.

## References

- L. Rudin, S. Osher, E. Fatemi, Nonlinear total variation based noise removal algorithms, Physica D, 1992.
- Y. Meyer, Oscillating Patterns in Image Processing and Nonlinear Evolution Equations, AMS University Lecture Series 22, 2001.
- D. Strong, T. Chan, Edge-preserving and scale-dependent properties of total variation regularization, Inverse Problems, 2003.
- A. Chambolle, An algorithm for total variation minimization and applications, JMIV, 2004.
- Y. Boykov, V. Kolmogorov, Computing geodesics and minimal surfaces via graph cuts, ICCV, 2003 (Cauchy-Crofton weights).

## License

MIT
