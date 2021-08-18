## Planar Homographies as Warp

A planar homography is an warp operation (which is a mapping from pixel coordinates from one camera frame to another) that makes a fundamental assumption of the points lying on a plane in the real world.

A homography **H** transforms one set of points (in homogeneous coordinates, like [x, y, 1]) to another set of points. (x1 ~= Hx2, where ~= represents identical to and ignore the scale factor).

Typically, we can obtain the corresponding point coordinates using feature detectors and feature matches. Then use them to calculate the homography.

Let x1 ~= Hx2 and H is a 3x3 matrix. We need four point pairs to get H solved becasue H gets 9 - 1 = 8 DoF (ignore the scaler factor). Then for each point pair, the relation can be 
rewritten as Aih=0 where h is a column vector reshaped from H, like [h11, h12, h13, ... h31, h32, h33]

Then given that x1 = [x1, y1, 1] and x2 = [x2, y2, 1], so that we can derive A like: [[x2, y2, 1, 0, 0, 0, -x1x2, -x1y2, -x1], [0, 0, 0, x2, y2, 1, -y1x2, -y1y2, -y1]].
Stack four pairs toghter then we can a 8x9 matrix A. ---> Ah = 0

We can use SVD to solve this Ah=0 where A = UΣV^T. Here U is a matrix of column vectors called the 'left singular vectors' and V is called the 'right singular vectors'. The matrix Σ is a diagonal matrix.

The columns of U are eigenvectors of AA^T and the ones of V are the eigenvectors of A^TA. We want to minimize the error in solution in the least-squares sense. Ideally, the product Ah should be 0 (when the last
diagonal element σ9 = 0). 

f(h) = 1/2 (Ah-0)^T (Ah - 0) = 1/2(Ah)^T(Ah) = 1/2·h^T·A^T·A·h, minimizing the error with respect to h using derivative, we get:

df/dh = 0 ---> 1/2·(A^TA + (A^TA)^T)h = 0 --> A^T·A·h = 0

This implies that the value of h equals the eigenvector corresponding to the zero eigenvalue (or closest to zero in case of noise). Then we should choose the smallest eigenvalue of A^T·A and it is the column 9 of matrix V.

## Homography with normalization

Normalization improves numerical stability of the solution and you should always normalize your coordinate data. This is a linear transformation and can be written as follows:

~x1 = T1x1  ~x2 = T2x2, where ~x1 and ~x2 are the normalized homogeneous coordinates of x1 and x2. Compute H from ~x1 = H~x2 and by substituting ~x1 and ~x2, we have:

x1 = T^-1·H·T2·x2
