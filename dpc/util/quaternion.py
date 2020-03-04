# Copyright Philipp Jund (jundp@cs.uni-freiburg.de) and Eldar Insafutdinov, 2018.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Website: https://github.com/PhilJd/tf-quaternion

import torch
import numpy as np
import torch.nn.functional as F

MAX_FLOAT = np.maximum_sctype(np.float)
FLOAT_EPS = np.finfo(np.float).eps


# TODO: Fix functions in this file. This path is not being taken in default train config. None of these functions are being called except last numpy function

def validate_shape(x):
    """Raise a value error if x.shape ist not (..., 4)."""
    error_msg = ("Can't create a quaternion from a tensor with shape {}."
                 "The last dimension must be 4.")
    # Check is performed during graph construction. If your dimension
    # is unknown, tf.reshape(x, (-1, 4)) might work.
    if x.shape[-1] != 4:
        raise ValueError(error_msg.format(x.shape))


def vector3d_to_quaternion(x):
    """Convert a tensor of 3D vectors to a quaternion.
    Prepends a 0 to the last dimension, i.e. [[1,2,3]] -> [[0,1,2,3]].
    Args:
        x: A `tf.Tensor` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
    Raises:
        ValueError, if the last dimension of x is not 3.
    """
    #x = torch.from_numpy(x)
    if x.shape[-1] != 3:
        raise ValueError("The last dimension of x must be 3.")
    padding = (1,0)
    return F.pad(x, padding)


def _prepare_tensor_for_div_mul(x):
    """Prepare the tensor x for division/multiplication.
    This function
    a) converts x to a tensor if necessary,
    b) prepends a 0 in the last dimension if the last dimension is 3,
    c) validates the type and shape.
    """
    #x = torch.from_numpy(x)
    if x.shape[-1] == 3:
        x = vector3d_to_quaternion(x)
    validate_shape(x)
    return x


def quaternion_multiply(a, b):
    """Multiply two quaternion tensors.
    Note that this differs from tf.multiply and is not commutative.
    Args:
        a, b: A `tf.Tensor` with shape (..., 4).
    Returns:
        A `Quaternion`.
    """
    a = _prepare_tensor_for_div_mul(a)
    b = _prepare_tensor_for_div_mul(b)
    w1, x1, y1, z1 = torch.unbind(a, dim=-1)
    w2, x2, y2, z2 = torch.unbind(b, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    stacked = torch.stack((w, x, y, z), dim=-1)
    return stacked


def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conj_array = torch.from_numpy(np.array([1.0, -1.0, -1.0, -1.0])).to(device)
    return q * conj_array


def quaternion_normalise(q):
    """Normalises quaternion to use as a rotation quaternion
    Args:
        q: [..., 4] quaternion
    Returns:
        q / ||q||_2
    """
    return q / q.norm(p=2,dim=-1).reshape(q.shape[0],1)


def quaternion_rotate(pc, q, inverse=False):
    """rotates a set of 3D points by a rotation,
    represented as a quaternion
    Args:
        pc: [B,N,3] point cloud
        q: [B,4] rotation quaternion
    Returns:
        q * pc * q'
    """
    q_norm = q.norm(p=2,dim=-1).reshape(q.shape[0],1)
    # TODO: Detach or not, the denominatior
    q = torch.div(q,q_norm)
    q = q.reshape(q.shape[0],1,q.shape[1])  # [B,1,4]
    q_ = quaternion_conjugate(q)
    qmul = quaternion_multiply
    if not inverse:
        wxyz = qmul(qmul(q, pc), q_)  # [B,N,4]
    else:
        wxyz = qmul(qmul(q_, pc), q)  # [B,N,4]
    if len(wxyz.shape) == 2: # bug with batch size of 1
        wxyz = torch.expand_dims(wxyz, axis=0)
    xyz = wxyz[:, :, 1:4]  # [B,N,3]
    return xyz


def normalized(q):
    q_norm = torch.norm(q, dim=-1).unsqueeze(-1)
    q /= q_norm
    return q


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
def as_rotation_matrix(q):
    """Calculate the corresponding rotation matrix.

    See
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/

    Returns:
        A `tf.Tensor` with R+1 dimensions and
        shape [d_1, ..., d_(R-1), 3, 3], the rotation matrix
    """
    # helper functions
    def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
        return 1 - 2 * torch.pow(a, 2) - 2 * torch.pow(b, 2)

    def tr_add(a, b, c, d):  # computes triangle entries with addition
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
        return 2 * a * b - 2 * c * d

        
    w, x, y, z = torch.split(normalized(q),1, dim=-1)
    m = [[diag(y, z).cpu().numpy(), tr_sub(x, y, z, w).cpu().numpy(), tr_add(x, z, y, w).cpu().numpy()],
         [tr_add(x, y, z, w).cpu().numpy(), diag(x, z).cpu().numpy(), tr_sub(y, z, x, w).cpu().numpy()],
         [tr_sub(x, z, y, w).cpu().numpy(), tr_add(y, z, x, w).cpu().numpy(), diag(x, y).cpu().numpy()]]
    return np.stack([np.stack(m[i], axis=-1) for i in range(3)], axis=-2)


def from_rotation_matrix(mtr):
    """
    See
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    mtr = torch.from_numpy(mtr)
    def m(j, i):
        shape = mtr.shape
        begin = [0 for _ in range(len(shape))]
        begin[-2] = j
        begin[-1] = i
        size = [s for s in shape]
        size[-2] = 1
        size[-1] = 1
        v = mtr[:,j:j+1,i:i+1]
        v = v.squeeze()
        return v

    w = torch.sqrt(1.0 + m(0, 0) + m(1, 1) + m(2, 2)) / 2
    x = (m(2, 1) - m(1, 2)) / (4 * w)
    y = (m(0, 2) - m(2, 0)) / (4 * w)
    z = (m(1, 0) - m(0, 1)) / (4 * w)
    q = torch.stack([w, x, y, z], dim=-1)
    return q

def mat2quat(M):
    ''' Calculate quaternion corresponding to given rotation matrix
    Method claimed to be robust to numerical errors in `M`.
    Constructs quaternion by calculating maximum eigenvector for matrix
    ``K`` (constructed from input `M`).  Although this is not tested, a maximum
    eigenvalue of 1 corresponds to a valid rotation.
    A quaternion ``q*-1`` corresponds to the same rotation as ``q``; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).
    See notes.
    Parameters
    ----------
    M : array-like
      3x3 rotation matrix
    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]
    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090
    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True
    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
    quaternion from a rotation matrix", AIAA Journal of Guidance,
    Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
    0731-5090
    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q
def quaternion_multiply_np(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])
