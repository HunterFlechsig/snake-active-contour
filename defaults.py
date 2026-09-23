"""Live-program hyperparameters. Kass states the energies, not these numbers."""

# Tension, bending, implicit stay-put weight, image-force scale.
ALPHA = 0.08
BETA = 0.40
GAMMA = 0.10
KAPPA = 1.00

# Image energy: edge term only, blur width 3.
W_LINE = 0.0
W_EDGE = 1.0
W_TERM = 0.0
SIGMA = 3.0

MAX_PX_MOVE = 1.0
SPRING_K = 0.08
VOLCANO_K = 900.0
VOLCANO_RADIUS = 20.0

AUTO_SIGMAS = (5.0, 3.0, 1.5)
AUTO_MAX_ITERATIONS = 800
AUTO_CONVERGENCE = 0.08

DENSITY = 0.5

# scikit-image astronaut example, used only when requested.
SKIMAGE_ALPHA = 0.015
SKIMAGE_BETA = 10.0
SKIMAGE_GAMMA = 0.001
