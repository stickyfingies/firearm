/**
 * Records what objects are colliding with each other
 */

/**
 * A 3x3 matrix.
 * Authored by {@link http://github.com/schteppe/ schteppe}
 */
class Mat3 {
  /**
   * A vector of length 9, containing all matrix elements.
   */

  /**
   * @param elements A vector of length 9, containing all matrix elements.
   */
  constructor(elements) {
    if (elements === void 0) {
      elements = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    }

    this.elements = elements;
  }
  /**
   * Sets the matrix to identity
   * @todo Should perhaps be renamed to `setIdentity()` to be more clear.
   * @todo Create another function that immediately creates an identity matrix eg. `eye()`
   */


  identity() {
    const e = this.elements;
    e[0] = 1;
    e[1] = 0;
    e[2] = 0;
    e[3] = 0;
    e[4] = 1;
    e[5] = 0;
    e[6] = 0;
    e[7] = 0;
    e[8] = 1;
  }
  /**
   * Set all elements to zero
   */


  setZero() {
    const e = this.elements;
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;
    e[3] = 0;
    e[4] = 0;
    e[5] = 0;
    e[6] = 0;
    e[7] = 0;
    e[8] = 0;
  }
  /**
   * Sets the matrix diagonal elements from a Vec3
   */


  setTrace(vector) {
    const e = this.elements;
    e[0] = vector.x;
    e[4] = vector.y;
    e[8] = vector.z;
  }
  /**
   * Gets the matrix diagonal elements
   */


  getTrace(target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const e = this.elements;
    target.x = e[0];
    target.y = e[4];
    target.z = e[8];
    return target;
  }
  /**
   * Matrix-Vector multiplication
   * @param v The vector to multiply with
   * @param target Optional, target to save the result in.
   */


  vmult(v, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const e = this.elements;
    const x = v.x;
    const y = v.y;
    const z = v.z;
    target.x = e[0] * x + e[1] * y + e[2] * z;
    target.y = e[3] * x + e[4] * y + e[5] * z;
    target.z = e[6] * x + e[7] * y + e[8] * z;
    return target;
  }
  /**
   * Matrix-scalar multiplication
   */


  smult(s) {
    for (let i = 0; i < this.elements.length; i++) {
      this.elements[i] *= s;
    }
  }
  /**
   * Matrix multiplication
   * @param matrix Matrix to multiply with from left side.
   */


  mmult(matrix, target) {
    if (target === void 0) {
      target = new Mat3();
    }

    const A = this.elements;
    const B = matrix.elements;
    const T = target.elements;
    const a11 = A[0],
          a12 = A[1],
          a13 = A[2],
          a21 = A[3],
          a22 = A[4],
          a23 = A[5],
          a31 = A[6],
          a32 = A[7],
          a33 = A[8];
    const b11 = B[0],
          b12 = B[1],
          b13 = B[2],
          b21 = B[3],
          b22 = B[4],
          b23 = B[5],
          b31 = B[6],
          b32 = B[7],
          b33 = B[8];
    T[0] = a11 * b11 + a12 * b21 + a13 * b31;
    T[1] = a11 * b12 + a12 * b22 + a13 * b32;
    T[2] = a11 * b13 + a12 * b23 + a13 * b33;
    T[3] = a21 * b11 + a22 * b21 + a23 * b31;
    T[4] = a21 * b12 + a22 * b22 + a23 * b32;
    T[5] = a21 * b13 + a22 * b23 + a23 * b33;
    T[6] = a31 * b11 + a32 * b21 + a33 * b31;
    T[7] = a31 * b12 + a32 * b22 + a33 * b32;
    T[8] = a31 * b13 + a32 * b23 + a33 * b33;
    return target;
  }
  /**
   * Scale each column of the matrix
   */


  scale(vector, target) {
    if (target === void 0) {
      target = new Mat3();
    }

    const e = this.elements;
    const t = target.elements;

    for (let i = 0; i !== 3; i++) {
      t[3 * i + 0] = vector.x * e[3 * i + 0];
      t[3 * i + 1] = vector.y * e[3 * i + 1];
      t[3 * i + 2] = vector.z * e[3 * i + 2];
    }

    return target;
  }
  /**
   * Solve Ax=b
   * @param b The right hand side
   * @param target Optional. Target vector to save in.
   * @return The solution x
   * @todo should reuse arrays
   */


  solve(b, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    // Construct equations
    const nr = 3; // num rows

    const nc = 4; // num cols

    const eqns = [];
    let i;
    let j;

    for (i = 0; i < nr * nc; i++) {
      eqns.push(0);
    }

    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        eqns[i + nc * j] = this.elements[i + 3 * j];
      }
    }

    eqns[3 + 4 * 0] = b.x;
    eqns[3 + 4 * 1] = b.y;
    eqns[3 + 4 * 2] = b.z; // Compute right upper triangular version of the matrix - Gauss elimination

    let n = 3;
    const k = n;
    let np;
    const kp = 4; // num rows

    let p;

    do {
      i = k - n;

      if (eqns[i + nc * i] === 0) {
        // the pivot is null, swap lines
        for (j = i + 1; j < k; j++) {
          if (eqns[i + nc * j] !== 0) {
            np = kp;

            do {
              // do ligne( i ) = ligne( i ) + ligne( k )
              p = kp - np;
              eqns[p + nc * i] += eqns[p + nc * j];
            } while (--np);

            break;
          }
        }
      }

      if (eqns[i + nc * i] !== 0) {
        for (j = i + 1; j < k; j++) {
          const multiplier = eqns[i + nc * j] / eqns[i + nc * i];
          np = kp;

          do {
            // do ligne( k ) = ligne( k ) - multiplier * ligne( i )
            p = kp - np;
            eqns[p + nc * j] = p <= i ? 0 : eqns[p + nc * j] - eqns[p + nc * i] * multiplier;
          } while (--np);
        }
      }
    } while (--n); // Get the solution


    target.z = eqns[2 * nc + 3] / eqns[2 * nc + 2];
    target.y = (eqns[1 * nc + 3] - eqns[1 * nc + 2] * target.z) / eqns[1 * nc + 1];
    target.x = (eqns[0 * nc + 3] - eqns[0 * nc + 2] * target.z - eqns[0 * nc + 1] * target.y) / eqns[0 * nc + 0];

    if (isNaN(target.x) || isNaN(target.y) || isNaN(target.z) || target.x === Infinity || target.y === Infinity || target.z === Infinity) {
      throw `Could not solve equation! Got x=[${target.toString()}], b=[${b.toString()}], A=[${this.toString()}]`;
    }

    return target;
  }
  /**
   * Get an element in the matrix by index. Index starts at 0, not 1!!!
   * @param value If provided, the matrix element will be set to this value.
   */


  e(row, column, value) {
    if (value === undefined) {
      return this.elements[column + 3 * row];
    } else {
      // Set value
      this.elements[column + 3 * row] = value;
    }
  }
  /**
   * Copy another matrix into this matrix object.
   */


  copy(matrix) {
    for (let i = 0; i < matrix.elements.length; i++) {
      this.elements[i] = matrix.elements[i];
    }

    return this;
  }
  /**
   * Returns a string representation of the matrix.
   */


  toString() {
    let r = '';
    const sep = ',';

    for (let i = 0; i < 9; i++) {
      r += this.elements[i] + sep;
    }

    return r;
  }
  /**
   * reverse the matrix
   * @param target Target matrix to save in.
   * @return The solution x
   */


  reverse(target) {
    if (target === void 0) {
      target = new Mat3();
    }

    // Construct equations
    const nr = 3; // num rows

    const nc = 6; // num cols

    const eqns = reverse_eqns;
    let i;
    let j;

    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        eqns[i + nc * j] = this.elements[i + 3 * j];
      }
    }

    eqns[3 + 6 * 0] = 1;
    eqns[3 + 6 * 1] = 0;
    eqns[3 + 6 * 2] = 0;
    eqns[4 + 6 * 0] = 0;
    eqns[4 + 6 * 1] = 1;
    eqns[4 + 6 * 2] = 0;
    eqns[5 + 6 * 0] = 0;
    eqns[5 + 6 * 1] = 0;
    eqns[5 + 6 * 2] = 1; // Compute right upper triangular version of the matrix - Gauss elimination

    let n = 3;
    const k = n;
    let np;
    const kp = nc; // num rows

    let p;

    do {
      i = k - n;

      if (eqns[i + nc * i] === 0) {
        // the pivot is null, swap lines
        for (j = i + 1; j < k; j++) {
          if (eqns[i + nc * j] !== 0) {
            np = kp;

            do {
              // do line( i ) = line( i ) + line( k )
              p = kp - np;
              eqns[p + nc * i] += eqns[p + nc * j];
            } while (--np);

            break;
          }
        }
      }

      if (eqns[i + nc * i] !== 0) {
        for (j = i + 1; j < k; j++) {
          const multiplier = eqns[i + nc * j] / eqns[i + nc * i];
          np = kp;

          do {
            // do line( k ) = line( k ) - multiplier * line( i )
            p = kp - np;
            eqns[p + nc * j] = p <= i ? 0 : eqns[p + nc * j] - eqns[p + nc * i] * multiplier;
          } while (--np);
        }
      }
    } while (--n); // eliminate the upper left triangle of the matrix


    i = 2;

    do {
      j = i - 1;

      do {
        const multiplier = eqns[i + nc * j] / eqns[i + nc * i];
        np = nc;

        do {
          p = nc - np;
          eqns[p + nc * j] = eqns[p + nc * j] - eqns[p + nc * i] * multiplier;
        } while (--np);
      } while (j--);
    } while (--i); // operations on the diagonal


    i = 2;

    do {
      const multiplier = 1 / eqns[i + nc * i];
      np = nc;

      do {
        p = nc - np;
        eqns[p + nc * i] = eqns[p + nc * i] * multiplier;
      } while (--np);
    } while (i--);

    i = 2;

    do {
      j = 2;

      do {
        p = eqns[nr + j + nc * i];

        if (isNaN(p) || p === Infinity) {
          throw `Could not reverse! A=[${this.toString()}]`;
        }

        target.e(i, j, p);
      } while (j--);
    } while (i--);

    return target;
  }
  /**
   * Set the matrix from a quaterion
   */


  setRotationFromQuaternion(q) {
    const x = q.x;
    const y = q.y;
    const z = q.z;
    const w = q.w;
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    const e = this.elements;
    e[3 * 0 + 0] = 1 - (yy + zz);
    e[3 * 0 + 1] = xy - wz;
    e[3 * 0 + 2] = xz + wy;
    e[3 * 1 + 0] = xy + wz;
    e[3 * 1 + 1] = 1 - (xx + zz);
    e[3 * 1 + 2] = yz - wx;
    e[3 * 2 + 0] = xz - wy;
    e[3 * 2 + 1] = yz + wx;
    e[3 * 2 + 2] = 1 - (xx + yy);
    return this;
  }
  /**
   * Transpose the matrix
   * @param target Optional. Where to store the result.
   * @return The target Mat3, or a new Mat3 if target was omitted.
   */


  transpose(target) {
    if (target === void 0) {
      target = new Mat3();
    }

    const M = this.elements;
    const T = target.elements;
    let tmp; //Set diagonals

    T[0] = M[0];
    T[4] = M[4];
    T[8] = M[8];
    tmp = M[1];
    T[1] = M[3];
    T[3] = tmp;
    tmp = M[2];
    T[2] = M[6];
    T[6] = tmp;
    tmp = M[5];
    T[5] = M[7];
    T[7] = tmp;
    return target;
  }

}
const reverse_eqns = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/**
 * 3-dimensional vector
 * @example
 *     const v = new Vec3(1, 2, 3)
 *     console.log('x=' + v.x) // x=1
 */

class Vec3 {
  constructor(x, y, z) {
    if (x === void 0) {
      x = 0.0;
    }

    if (y === void 0) {
      y = 0.0;
    }

    if (z === void 0) {
      z = 0.0;
    }

    this.x = x;
    this.y = y;
    this.z = z;
  }
  /**
   * Vector cross product
   * @param target Optional target to save in.
   */


  cross(vector, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const vx = vector.x;
    const vy = vector.y;
    const vz = vector.z;
    const x = this.x;
    const y = this.y;
    const z = this.z;
    target.x = y * vz - z * vy;
    target.y = z * vx - x * vz;
    target.z = x * vy - y * vx;
    return target;
  }
  /**
   * Set the vectors' 3 elements
   */


  set(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  /**
   * Set all components of the vector to zero.
   */


  setZero() {
    this.x = this.y = this.z = 0;
  }
  /**
   * Vector addition
   */


  vadd(vector, target) {
    if (target) {
      target.x = vector.x + this.x;
      target.y = vector.y + this.y;
      target.z = vector.z + this.z;
    } else {
      return new Vec3(this.x + vector.x, this.y + vector.y, this.z + vector.z);
    }
  }
  /**
   * Vector subtraction
   * @param target Optional target to save in.
   */


  vsub(vector, target) {
    if (target) {
      target.x = this.x - vector.x;
      target.y = this.y - vector.y;
      target.z = this.z - vector.z;
    } else {
      return new Vec3(this.x - vector.x, this.y - vector.y, this.z - vector.z);
    }
  }
  /**
   * Get the cross product matrix a_cross from a vector, such that a x b = a_cross * b = c
   *
   * See {@link https://www8.cs.umu.se/kurser/TDBD24/VT06/lectures/Lecture6.pdf Umeå University Lecture}
   */


  crossmat() {
    return new Mat3([0, -this.z, this.y, this.z, 0, -this.x, -this.y, this.x, 0]);
  }
  /**
   * Normalize the vector. Note that this changes the values in the vector.
    * @return Returns the norm of the vector
   */


  normalize() {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const n = Math.sqrt(x * x + y * y + z * z);

    if (n > 0.0) {
      const invN = 1 / n;
      this.x *= invN;
      this.y *= invN;
      this.z *= invN;
    } else {
      // Make something up
      this.x = 0;
      this.y = 0;
      this.z = 0;
    }

    return n;
  }
  /**
   * Get the version of this vector that is of length 1.
   * @param target Optional target to save in
   * @return Returns the unit vector
   */


  unit(target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const x = this.x;
    const y = this.y;
    const z = this.z;
    let ninv = Math.sqrt(x * x + y * y + z * z);

    if (ninv > 0.0) {
      ninv = 1.0 / ninv;
      target.x = x * ninv;
      target.y = y * ninv;
      target.z = z * ninv;
    } else {
      target.x = 1;
      target.y = 0;
      target.z = 0;
    }

    return target;
  }
  /**
   * Get the length of the vector
   */


  length() {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    return Math.sqrt(x * x + y * y + z * z);
  }
  /**
   * Get the squared length of the vector.
   */


  lengthSquared() {
    return this.dot(this);
  }
  /**
   * Get distance from this point to another point
   */


  distanceTo(p) {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const px = p.x;
    const py = p.y;
    const pz = p.z;
    return Math.sqrt((px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z));
  }
  /**
   * Get squared distance from this point to another point
   */


  distanceSquared(p) {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const px = p.x;
    const py = p.y;
    const pz = p.z;
    return (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z);
  }
  /**
   * Multiply all the components of the vector with a scalar.
   * @param target The vector to save the result in.
   */


  scale(scalar, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const x = this.x;
    const y = this.y;
    const z = this.z;
    target.x = scalar * x;
    target.y = scalar * y;
    target.z = scalar * z;
    return target;
  }
  /**
   * Multiply the vector with an other vector, component-wise.
   * @param target The vector to save the result in.
   */


  vmul(vector, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    target.x = vector.x * this.x;
    target.y = vector.y * this.y;
    target.z = vector.z * this.z;
    return target;
  }
  /**
   * Scale a vector and add it to this vector. Save the result in "target". (target = this + vector * scalar)
   * @param target The vector to save the result in.
   */


  addScaledVector(scalar, vector, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    target.x = this.x + scalar * vector.x;
    target.y = this.y + scalar * vector.y;
    target.z = this.z + scalar * vector.z;
    return target;
  }
  /**
   * Calculate dot product
   * @param vector
   */


  dot(vector) {
    return this.x * vector.x + this.y * vector.y + this.z * vector.z;
  }

  isZero() {
    return this.x === 0 && this.y === 0 && this.z === 0;
  }
  /**
   * Make the vector point in the opposite direction.
   * @param target Optional target to save in
   */


  negate(target) {
    if (target === void 0) {
      target = new Vec3();
    }

    target.x = -this.x;
    target.y = -this.y;
    target.z = -this.z;
    return target;
  }
  /**
   * Compute two artificial tangents to the vector
   * @param t1 Vector object to save the first tangent in
   * @param t2 Vector object to save the second tangent in
   */


  tangents(t1, t2) {
    const norm = this.length();

    if (norm > 0.0) {
      const n = Vec3_tangents_n;
      const inorm = 1 / norm;
      n.set(this.x * inorm, this.y * inorm, this.z * inorm);
      const randVec = Vec3_tangents_randVec;

      if (Math.abs(n.x) < 0.9) {
        randVec.set(1, 0, 0);
        n.cross(randVec, t1);
      } else {
        randVec.set(0, 1, 0);
        n.cross(randVec, t1);
      }

      n.cross(t1, t2);
    } else {
      // The normal length is zero, make something up
      t1.set(1, 0, 0);
      t2.set(0, 1, 0);
    }
  }
  /**
   * Converts to a more readable format
   */


  toString() {
    return `${this.x},${this.y},${this.z}`;
  }
  /**
   * Converts to an array
   */


  toArray() {
    return [this.x, this.y, this.z];
  }
  /**
   * Copies value of source to this vector.
   */


  copy(vector) {
    this.x = vector.x;
    this.y = vector.y;
    this.z = vector.z;
    return this;
  }
  /**
   * Do a linear interpolation between two vectors
   * @param t A number between 0 and 1. 0 will make this function return u, and 1 will make it return v. Numbers in between will generate a vector in between them.
   */


  lerp(vector, t, target) {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    target.x = x + (vector.x - x) * t;
    target.y = y + (vector.y - y) * t;
    target.z = z + (vector.z - z) * t;
  }
  /**
   * Check if a vector equals is almost equal to another one.
   */


  almostEquals(vector, precision) {
    if (precision === void 0) {
      precision = 1e-6;
    }

    if (Math.abs(this.x - vector.x) > precision || Math.abs(this.y - vector.y) > precision || Math.abs(this.z - vector.z) > precision) {
      return false;
    }

    return true;
  }
  /**
   * Check if a vector is almost zero
   */


  almostZero(precision) {
    if (precision === void 0) {
      precision = 1e-6;
    }

    if (Math.abs(this.x) > precision || Math.abs(this.y) > precision || Math.abs(this.z) > precision) {
      return false;
    }

    return true;
  }
  /**
   * Check if the vector is anti-parallel to another vector.
   * @param precision Set to zero for exact comparisons
   */


  isAntiparallelTo(vector, precision) {
    this.negate(antip_neg);
    return antip_neg.almostEquals(vector, precision);
  }
  /**
   * Clone the vector
   */


  clone() {
    return new Vec3(this.x, this.y, this.z);
  }

}
Vec3.ZERO = new Vec3(0, 0, 0);
Vec3.UNIT_X = new Vec3(1, 0, 0);
Vec3.UNIT_Y = new Vec3(0, 1, 0);
Vec3.UNIT_Z = new Vec3(0, 0, 1);
const Vec3_tangents_n = new Vec3();
const Vec3_tangents_randVec = new Vec3();
const antip_neg = new Vec3();

/**
 * Axis aligned bounding box class.
 */
class AABB {
  /**
   * The lower bound of the bounding box
   */

  /**
   * The upper bound of the bounding box
   */
  constructor(options) {
    if (options === void 0) {
      options = {};
    }

    this.lowerBound = new Vec3();
    this.upperBound = new Vec3();

    if (options.lowerBound) {
      this.lowerBound.copy(options.lowerBound);
    }

    if (options.upperBound) {
      this.upperBound.copy(options.upperBound);
    }
  }
  /**
   * Set the AABB bounds from a set of points.
   * @param points An array of Vec3's.
   * @return The self object
   */


  setFromPoints(points, position, quaternion, skinSize) {
    const l = this.lowerBound;
    const u = this.upperBound;
    const q = quaternion; // Set to the first point

    l.copy(points[0]);

    if (q) {
      q.vmult(l, l);
    }

    u.copy(l);

    for (let i = 1; i < points.length; i++) {
      let p = points[i];

      if (q) {
        q.vmult(p, tmp$1);
        p = tmp$1;
      }

      if (p.x > u.x) {
        u.x = p.x;
      }

      if (p.x < l.x) {
        l.x = p.x;
      }

      if (p.y > u.y) {
        u.y = p.y;
      }

      if (p.y < l.y) {
        l.y = p.y;
      }

      if (p.z > u.z) {
        u.z = p.z;
      }

      if (p.z < l.z) {
        l.z = p.z;
      }
    } // Add offset


    if (position) {
      position.vadd(l, l);
      position.vadd(u, u);
    }

    if (skinSize) {
      l.x -= skinSize;
      l.y -= skinSize;
      l.z -= skinSize;
      u.x += skinSize;
      u.y += skinSize;
      u.z += skinSize;
    }

    return this;
  }
  /**
   * Copy bounds from an AABB to this AABB
   * @param aabb Source to copy from
   * @return The this object, for chainability
   */


  copy(aabb) {
    this.lowerBound.copy(aabb.lowerBound);
    this.upperBound.copy(aabb.upperBound);
    return this;
  }
  /**
   * Clone an AABB
   */


  clone() {
    return new AABB().copy(this);
  }
  /**
   * Extend this AABB so that it covers the given AABB too.
   */


  extend(aabb) {
    this.lowerBound.x = Math.min(this.lowerBound.x, aabb.lowerBound.x);
    this.upperBound.x = Math.max(this.upperBound.x, aabb.upperBound.x);
    this.lowerBound.y = Math.min(this.lowerBound.y, aabb.lowerBound.y);
    this.upperBound.y = Math.max(this.upperBound.y, aabb.upperBound.y);
    this.lowerBound.z = Math.min(this.lowerBound.z, aabb.lowerBound.z);
    this.upperBound.z = Math.max(this.upperBound.z, aabb.upperBound.z);
  }
  /**
   * Returns true if the given AABB overlaps this AABB.
   */


  overlaps(aabb) {
    const l1 = this.lowerBound;
    const u1 = this.upperBound;
    const l2 = aabb.lowerBound;
    const u2 = aabb.upperBound; //      l2        u2
    //      |---------|
    // |--------|
    // l1       u1

    const overlapsX = l2.x <= u1.x && u1.x <= u2.x || l1.x <= u2.x && u2.x <= u1.x;
    const overlapsY = l2.y <= u1.y && u1.y <= u2.y || l1.y <= u2.y && u2.y <= u1.y;
    const overlapsZ = l2.z <= u1.z && u1.z <= u2.z || l1.z <= u2.z && u2.z <= u1.z;
    return overlapsX && overlapsY && overlapsZ;
  } // Mostly for debugging


  volume() {
    const l = this.lowerBound;
    const u = this.upperBound;
    return (u.x - l.x) * (u.y - l.y) * (u.z - l.z);
  }
  /**
   * Returns true if the given AABB is fully contained in this AABB.
   */


  contains(aabb) {
    const l1 = this.lowerBound;
    const u1 = this.upperBound;
    const l2 = aabb.lowerBound;
    const u2 = aabb.upperBound; //      l2        u2
    //      |---------|
    // |---------------|
    // l1              u1

    return l1.x <= l2.x && u1.x >= u2.x && l1.y <= l2.y && u1.y >= u2.y && l1.z <= l2.z && u1.z >= u2.z;
  }

  getCorners(a, b, c, d, e, f, g, h) {
    const l = this.lowerBound;
    const u = this.upperBound;
    a.copy(l);
    b.set(u.x, l.y, l.z);
    c.set(u.x, u.y, l.z);
    d.set(l.x, u.y, u.z);
    e.set(u.x, l.y, u.z);
    f.set(l.x, u.y, l.z);
    g.set(l.x, l.y, u.z);
    h.copy(u);
  }
  /**
   * Get the representation of an AABB in another frame.
   * @return The "target" AABB object.
   */


  toLocalFrame(frame, target) {
    const corners = transformIntoFrame_corners;
    const a = corners[0];
    const b = corners[1];
    const c = corners[2];
    const d = corners[3];
    const e = corners[4];
    const f = corners[5];
    const g = corners[6];
    const h = corners[7]; // Get corners in current frame

    this.getCorners(a, b, c, d, e, f, g, h); // Transform them to new local frame

    for (let i = 0; i !== 8; i++) {
      const corner = corners[i];
      frame.pointToLocal(corner, corner);
    }

    return target.setFromPoints(corners);
  }
  /**
   * Get the representation of an AABB in the global frame.
   * @return The "target" AABB object.
   */


  toWorldFrame(frame, target) {
    const corners = transformIntoFrame_corners;
    const a = corners[0];
    const b = corners[1];
    const c = corners[2];
    const d = corners[3];
    const e = corners[4];
    const f = corners[5];
    const g = corners[6];
    const h = corners[7]; // Get corners in current frame

    this.getCorners(a, b, c, d, e, f, g, h); // Transform them to new local frame

    for (let i = 0; i !== 8; i++) {
      const corner = corners[i];
      frame.pointToWorld(corner, corner);
    }

    return target.setFromPoints(corners);
  }
  /**
   * Check if the AABB is hit by a ray.
   */


  overlapsRay(ray) {
    const {
      direction,
      from
    } = ray; // const t = 0
    // ray.direction is unit direction vector of ray

    const dirFracX = 1 / direction.x;
    const dirFracY = 1 / direction.y;
    const dirFracZ = 1 / direction.z; // this.lowerBound is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner

    const t1 = (this.lowerBound.x - from.x) * dirFracX;
    const t2 = (this.upperBound.x - from.x) * dirFracX;
    const t3 = (this.lowerBound.y - from.y) * dirFracY;
    const t4 = (this.upperBound.y - from.y) * dirFracY;
    const t5 = (this.lowerBound.z - from.z) * dirFracZ;
    const t6 = (this.upperBound.z - from.z) * dirFracZ; // const tmin = Math.max(Math.max(Math.min(t1, t2), Math.min(t3, t4)));
    // const tmax = Math.min(Math.min(Math.max(t1, t2), Math.max(t3, t4)));

    const tmin = Math.max(Math.max(Math.min(t1, t2), Math.min(t3, t4)), Math.min(t5, t6));
    const tmax = Math.min(Math.min(Math.max(t1, t2), Math.max(t3, t4)), Math.max(t5, t6)); // if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us

    if (tmax < 0) {
      //t = tmax;
      return false;
    } // if tmin > tmax, ray doesn't intersect AABB


    if (tmin > tmax) {
      //t = tmax;
      return false;
    }

    return true;
  }

}
const tmp$1 = new Vec3();
const transformIntoFrame_corners = [new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3()];

/**
 * Base class for objects that dispatches events.
 */
class EventTarget {
  /**
   * Add an event listener
   * @return The self object, for chainability.
   */
  addEventListener(type, listener) {
    if (this._listeners === undefined) {
      this._listeners = {};
    }

    const listeners = this._listeners;

    if (listeners[type] === undefined) {
      listeners[type] = [];
    }

    if (!listeners[type].includes(listener)) {
      listeners[type].push(listener);
    }

    return this;
  }
  /**
   * Check if an event listener is added
   */


  hasEventListener(type, listener) {
    if (this._listeners === undefined) {
      return false;
    }

    const listeners = this._listeners;

    if (listeners[type] !== undefined && listeners[type].includes(listener)) {
      return true;
    }

    return false;
  }
  /**
   * Check if any event listener of the given type is added
   */


  hasAnyEventListener(type) {
    if (this._listeners === undefined) {
      return false;
    }

    const listeners = this._listeners;
    return listeners[type] !== undefined;
  }
  /**
   * Remove an event listener
   * @return The self object, for chainability.
   */


  removeEventListener(type, listener) {
    if (this._listeners === undefined) {
      return this;
    }

    const listeners = this._listeners;

    if (listeners[type] === undefined) {
      return this;
    }

    const index = listeners[type].indexOf(listener);

    if (index !== -1) {
      listeners[type].splice(index, 1);
    }

    return this;
  }
  /**
   * Emit an event.
   * @return The self object, for chainability.
   */


  dispatchEvent(event) {
    if (this._listeners === undefined) {
      return this;
    }

    const listeners = this._listeners;
    const listenerArray = listeners[event.type];

    if (listenerArray !== undefined) {
      event.target = this;

      for (let i = 0, l = listenerArray.length; i < l; i++) {
        listenerArray[i].call(this, event);
      }
    }

    return this;
  }

}

/**
 * A Quaternion describes a rotation in 3D space. The Quaternion is mathematically defined as Q = x*i + y*j + z*k + w, where (i,j,k) are imaginary basis vectors. (x,y,z) can be seen as a vector related to the axis of rotation, while the real multiplier, w, is related to the amount of rotation.
 * @param x Multiplier of the imaginary basis vector i.
 * @param y Multiplier of the imaginary basis vector j.
 * @param z Multiplier of the imaginary basis vector k.
 * @param w Multiplier of the real part.
 * @see http://en.wikipedia.org/wiki/Quaternion
 */

class Quaternion {
  constructor(x, y, z, w) {
    if (x === void 0) {
      x = 0;
    }

    if (y === void 0) {
      y = 0;
    }

    if (z === void 0) {
      z = 0;
    }

    if (w === void 0) {
      w = 1;
    }

    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }
  /**
   * Set the value of the quaternion.
   */


  set(x, y, z, w) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    return this;
  }
  /**
   * Convert to a readable format
   * @return "x,y,z,w"
   */


  toString() {
    return `${this.x},${this.y},${this.z},${this.w}`;
  }
  /**
   * Convert to an Array
   * @return [x, y, z, w]
   */


  toArray() {
    return [this.x, this.y, this.z, this.w];
  }
  /**
   * Set the quaternion components given an axis and an angle in radians.
   */


  setFromAxisAngle(vector, angle) {
    const s = Math.sin(angle * 0.5);
    this.x = vector.x * s;
    this.y = vector.y * s;
    this.z = vector.z * s;
    this.w = Math.cos(angle * 0.5);
    return this;
  }
  /**
   * Converts the quaternion to [ axis, angle ] representation.
   * @param targetAxis A vector object to reuse for storing the axis.
   * @return An array, first element is the axis and the second is the angle in radians.
   */


  toAxisAngle(targetAxis) {
    if (targetAxis === void 0) {
      targetAxis = new Vec3();
    }

    this.normalize(); // if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised

    const angle = 2 * Math.acos(this.w);
    const s = Math.sqrt(1 - this.w * this.w); // assuming quaternion normalised then w is less than 1, so term always positive.

    if (s < 0.001) {
      // test to avoid divide by zero, s is always positive due to sqrt
      // if s close to zero then direction of axis not important
      targetAxis.x = this.x; // if it is important that axis is normalised then replace with x=1; y=z=0;

      targetAxis.y = this.y;
      targetAxis.z = this.z;
    } else {
      targetAxis.x = this.x / s; // normalise axis

      targetAxis.y = this.y / s;
      targetAxis.z = this.z / s;
    }

    return [targetAxis, angle];
  }
  /**
   * Set the quaternion value given two vectors. The resulting rotation will be the needed rotation to rotate u to v.
   */


  setFromVectors(u, v) {
    if (u.isAntiparallelTo(v)) {
      const t1 = sfv_t1;
      const t2 = sfv_t2;
      u.tangents(t1, t2);
      this.setFromAxisAngle(t1, Math.PI);
    } else {
      const a = u.cross(v);
      this.x = a.x;
      this.y = a.y;
      this.z = a.z;
      this.w = Math.sqrt(u.length() ** 2 * v.length() ** 2) + u.dot(v);
      this.normalize();
    }

    return this;
  }
  /**
   * Multiply the quaternion with an other quaternion.
   */


  mult(quat, target) {
    if (target === void 0) {
      target = new Quaternion();
    }

    const ax = this.x;
    const ay = this.y;
    const az = this.z;
    const aw = this.w;
    const bx = quat.x;
    const by = quat.y;
    const bz = quat.z;
    const bw = quat.w;
    target.x = ax * bw + aw * bx + ay * bz - az * by;
    target.y = ay * bw + aw * by + az * bx - ax * bz;
    target.z = az * bw + aw * bz + ax * by - ay * bx;
    target.w = aw * bw - ax * bx - ay * by - az * bz;
    return target;
  }
  /**
   * Get the inverse quaternion rotation.
   */


  inverse(target) {
    if (target === void 0) {
      target = new Quaternion();
    }

    const x = this.x;
    const y = this.y;
    const z = this.z;
    const w = this.w;
    this.conjugate(target);
    const inorm2 = 1 / (x * x + y * y + z * z + w * w);
    target.x *= inorm2;
    target.y *= inorm2;
    target.z *= inorm2;
    target.w *= inorm2;
    return target;
  }
  /**
   * Get the quaternion conjugate
   */


  conjugate(target) {
    if (target === void 0) {
      target = new Quaternion();
    }

    target.x = -this.x;
    target.y = -this.y;
    target.z = -this.z;
    target.w = this.w;
    return target;
  }
  /**
   * Normalize the quaternion. Note that this changes the values of the quaternion.
   */


  normalize() {
    let l = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);

    if (l === 0) {
      this.x = 0;
      this.y = 0;
      this.z = 0;
      this.w = 0;
    } else {
      l = 1 / l;
      this.x *= l;
      this.y *= l;
      this.z *= l;
      this.w *= l;
    }

    return this;
  }
  /**
   * Approximation of quaternion normalization. Works best when quat is already almost-normalized.
   * @author unphased, https://github.com/unphased
   */


  normalizeFast() {
    const f = (3.0 - (this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w)) / 2.0;

    if (f === 0) {
      this.x = 0;
      this.y = 0;
      this.z = 0;
      this.w = 0;
    } else {
      this.x *= f;
      this.y *= f;
      this.z *= f;
      this.w *= f;
    }

    return this;
  }
  /**
   * Multiply the quaternion by a vector
   */


  vmult(v, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const x = v.x;
    const y = v.y;
    const z = v.z;
    const qx = this.x;
    const qy = this.y;
    const qz = this.z;
    const qw = this.w; // q*v

    const ix = qw * x + qy * z - qz * y;
    const iy = qw * y + qz * x - qx * z;
    const iz = qw * z + qx * y - qy * x;
    const iw = -qx * x - qy * y - qz * z;
    target.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    target.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    target.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;
    return target;
  }
  /**
   * Copies value of source to this quaternion.
   * @return this
   */


  copy(quat) {
    this.x = quat.x;
    this.y = quat.y;
    this.z = quat.z;
    this.w = quat.w;
    return this;
  }
  /**
   * Convert the quaternion to euler angle representation. Order: YZX, as this page describes: https://www.euclideanspace.com/maths/standards/index.htm
   * @param order Three-character string, defaults to "YZX"
   */


  toEuler(target, order) {
    if (order === void 0) {
      order = 'YZX';
    }

    let heading;
    let attitude;
    let bank;
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const w = this.w;

    switch (order) {
      case 'YZX':
        const test = x * y + z * w;

        if (test > 0.499) {
          // singularity at north pole
          heading = 2 * Math.atan2(x, w);
          attitude = Math.PI / 2;
          bank = 0;
        }

        if (test < -0.499) {
          // singularity at south pole
          heading = -2 * Math.atan2(x, w);
          attitude = -Math.PI / 2;
          bank = 0;
        }

        if (heading === undefined) {
          const sqx = x * x;
          const sqy = y * y;
          const sqz = z * z;
          heading = Math.atan2(2 * y * w - 2 * x * z, 1 - 2 * sqy - 2 * sqz); // Heading

          attitude = Math.asin(2 * test); // attitude

          bank = Math.atan2(2 * x * w - 2 * y * z, 1 - 2 * sqx - 2 * sqz); // bank
        }

        break;

      default:
        throw new Error(`Euler order ${order} not supported yet.`);
    }

    target.y = heading;
    target.z = attitude;
    target.x = bank;
  }
  /**
   * @param order The order to apply angles: 'XYZ' or 'YXZ' or any other combination.
   *
   * See {@link https://www.mathworks.com/matlabcentral/fileexchange/20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors MathWorks} reference
   */


  setFromEuler(x, y, z, order) {
    if (order === void 0) {
      order = 'XYZ';
    }

    const c1 = Math.cos(x / 2);
    const c2 = Math.cos(y / 2);
    const c3 = Math.cos(z / 2);
    const s1 = Math.sin(x / 2);
    const s2 = Math.sin(y / 2);
    const s3 = Math.sin(z / 2);

    if (order === 'XYZ') {
      this.x = s1 * c2 * c3 + c1 * s2 * s3;
      this.y = c1 * s2 * c3 - s1 * c2 * s3;
      this.z = c1 * c2 * s3 + s1 * s2 * c3;
      this.w = c1 * c2 * c3 - s1 * s2 * s3;
    } else if (order === 'YXZ') {
      this.x = s1 * c2 * c3 + c1 * s2 * s3;
      this.y = c1 * s2 * c3 - s1 * c2 * s3;
      this.z = c1 * c2 * s3 - s1 * s2 * c3;
      this.w = c1 * c2 * c3 + s1 * s2 * s3;
    } else if (order === 'ZXY') {
      this.x = s1 * c2 * c3 - c1 * s2 * s3;
      this.y = c1 * s2 * c3 + s1 * c2 * s3;
      this.z = c1 * c2 * s3 + s1 * s2 * c3;
      this.w = c1 * c2 * c3 - s1 * s2 * s3;
    } else if (order === 'ZYX') {
      this.x = s1 * c2 * c3 - c1 * s2 * s3;
      this.y = c1 * s2 * c3 + s1 * c2 * s3;
      this.z = c1 * c2 * s3 - s1 * s2 * c3;
      this.w = c1 * c2 * c3 + s1 * s2 * s3;
    } else if (order === 'YZX') {
      this.x = s1 * c2 * c3 + c1 * s2 * s3;
      this.y = c1 * s2 * c3 + s1 * c2 * s3;
      this.z = c1 * c2 * s3 - s1 * s2 * c3;
      this.w = c1 * c2 * c3 - s1 * s2 * s3;
    } else if (order === 'XZY') {
      this.x = s1 * c2 * c3 - c1 * s2 * s3;
      this.y = c1 * s2 * c3 - s1 * c2 * s3;
      this.z = c1 * c2 * s3 + s1 * s2 * c3;
      this.w = c1 * c2 * c3 + s1 * s2 * s3;
    }

    return this;
  }

  clone() {
    return new Quaternion(this.x, this.y, this.z, this.w);
  }
  /**
   * Performs a spherical linear interpolation between two quat
   *
   * @param toQuat second operand
   * @param t interpolation amount between the self quaternion and toQuat
   * @param target A quaternion to store the result in. If not provided, a new one will be created.
   * @returns {Quaternion} The "target" object
   */


  slerp(toQuat, t, target) {
    if (target === void 0) {
      target = new Quaternion();
    }

    const ax = this.x;
    const ay = this.y;
    const az = this.z;
    const aw = this.w;
    let bx = toQuat.x;
    let by = toQuat.y;
    let bz = toQuat.z;
    let bw = toQuat.w;
    let omega;
    let cosom;
    let sinom;
    let scale0;
    let scale1; // calc cosine

    cosom = ax * bx + ay * by + az * bz + aw * bw; // adjust signs (if necessary)

    if (cosom < 0.0) {
      cosom = -cosom;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    } // calculate coefficients


    if (1.0 - cosom > 0.000001) {
      // standard case (slerp)
      omega = Math.acos(cosom);
      sinom = Math.sin(omega);
      scale0 = Math.sin((1.0 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      // "from" and "to" quaternions are very close
      //  ... so we can do a linear interpolation
      scale0 = 1.0 - t;
      scale1 = t;
    } // calculate final values


    target.x = scale0 * ax + scale1 * bx;
    target.y = scale0 * ay + scale1 * by;
    target.z = scale0 * az + scale1 * bz;
    target.w = scale0 * aw + scale1 * bw;
    return target;
  }
  /**
   * Rotate an absolute orientation quaternion given an angular velocity and a time step.
   */


  integrate(angularVelocity, dt, angularFactor, target) {
    if (target === void 0) {
      target = new Quaternion();
    }

    const ax = angularVelocity.x * angularFactor.x,
          ay = angularVelocity.y * angularFactor.y,
          az = angularVelocity.z * angularFactor.z,
          bx = this.x,
          by = this.y,
          bz = this.z,
          bw = this.w;
    const half_dt = dt * 0.5;
    target.x += half_dt * (ax * bw + ay * bz - az * by);
    target.y += half_dt * (ay * bw + az * bx - ax * bz);
    target.z += half_dt * (az * bw + ax * by - ay * bx);
    target.w += half_dt * (-ax * bx - ay * by - az * bz);
    return target;
  }

}
const sfv_t1 = new Vec3();
const sfv_t2 = new Vec3();

/**
 * The available shape types.
 */
const SHAPE_TYPES = {
  /** SPHERE */
  SPHERE: 1,

  /** PLANE */
  PLANE: 2,

  /** BOX */
  BOX: 4,

  /** COMPOUND */
  COMPOUND: 8,

  /** CONVEXPOLYHEDRON */
  CONVEXPOLYHEDRON: 16,

  /** HEIGHTFIELD */
  HEIGHTFIELD: 32,

  /** PARTICLE */
  PARTICLE: 64,

  /** CYLINDER */
  CYLINDER: 128,

  /** TRIMESH */
  TRIMESH: 256
};
/**
 * ShapeType
 */

/**
 * Base class for shapes
 */
class Shape {
  /**
   * Identifier of the Shape.
   */

  /**
   * The type of this shape. Must be set to an int > 0 by subclasses.
   */

  /**
   * The local bounding sphere radius of this shape.
   */

  /**
   * Whether to produce contact forces when in contact with other bodies. Note that contacts will be generated, but they will be disabled.
   * @default true
   */

  /**
   * @default 1
   */

  /**
   * @default -1
   */

  /**
   * Optional material of the shape that regulates contact properties.
   */

  /**
   * The body to which the shape is added to.
   */

  /**
   * All the Shape types.
   */
  constructor(options) {
    if (options === void 0) {
      options = {};
    }

    this.id = Shape.idCounter++;
    this.type = options.type || 0;
    this.boundingSphereRadius = 0;
    this.collisionResponse = options.collisionResponse ? options.collisionResponse : true;
    this.collisionFilterGroup = options.collisionFilterGroup !== undefined ? options.collisionFilterGroup : 1;
    this.collisionFilterMask = options.collisionFilterMask !== undefined ? options.collisionFilterMask : -1;
    this.material = options.material ? options.material : null;
    this.body = null;
  }
  /**
   * Computes the bounding sphere radius.
   * The result is stored in the property `.boundingSphereRadius`
   */


  updateBoundingSphereRadius() {
    throw `computeBoundingSphereRadius() not implemented for shape type ${this.type}`;
  }
  /**
   * Get the volume of this shape
   */


  volume() {
    throw `volume() not implemented for shape type ${this.type}`;
  }
  /**
   * Calculates the inertia in the local frame for this shape.
   * @see http://en.wikipedia.org/wiki/List_of_moments_of_inertia
   */


  calculateLocalInertia(mass, target) {
    throw `calculateLocalInertia() not implemented for shape type ${this.type}`;
  }
  /**
   * @todo use abstract for these kind of methods
   */


  calculateWorldAABB(pos, quat, min, max) {
    throw `calculateWorldAABB() not implemented for shape type ${this.type}`;
  }

}
Shape.idCounter = 0;
Shape.types = SHAPE_TYPES;

/**
 * Transformation utilities.
 */
class Transform {
  /**
   * position
   */

  /**
   * quaternion
   */
  constructor(options) {
    if (options === void 0) {
      options = {};
    }

    this.position = new Vec3();
    this.quaternion = new Quaternion();

    if (options.position) {
      this.position.copy(options.position);
    }

    if (options.quaternion) {
      this.quaternion.copy(options.quaternion);
    }
  }
  /**
   * Get a global point in local transform coordinates.
   */


  pointToLocal(worldPoint, result) {
    return Transform.pointToLocalFrame(this.position, this.quaternion, worldPoint, result);
  }
  /**
   * Get a local point in global transform coordinates.
   */


  pointToWorld(localPoint, result) {
    return Transform.pointToWorldFrame(this.position, this.quaternion, localPoint, result);
  }
  /**
   * vectorToWorldFrame
   */


  vectorToWorldFrame(localVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    this.quaternion.vmult(localVector, result);
    return result;
  }
  /**
   * pointToLocalFrame
   */


  static pointToLocalFrame(position, quaternion, worldPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    worldPoint.vsub(position, result);
    quaternion.conjugate(tmpQuat$1);
    tmpQuat$1.vmult(result, result);
    return result;
  }
  /**
   * pointToWorldFrame
   */


  static pointToWorldFrame(position, quaternion, localPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    quaternion.vmult(localPoint, result);
    result.vadd(position, result);
    return result;
  }
  /**
   * vectorToWorldFrame
   */


  static vectorToWorldFrame(quaternion, localVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    quaternion.vmult(localVector, result);
    return result;
  }
  /**
   * vectorToLocalFrame
   */


  static vectorToLocalFrame(position, quaternion, worldVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    quaternion.w *= -1;
    quaternion.vmult(worldVector, result);
    quaternion.w *= -1;
    return result;
  }

}
const tmpQuat$1 = new Quaternion();

/**
 * A set of polygons describing a convex shape.
 *
 * The shape MUST be convex for the code to work properly. No polygons may be coplanar (contained
 * in the same 3D plane), instead these should be merged into one polygon.
 *
 * @author qiao / https://github.com/qiao (original author, see https://github.com/qiao/three.js/commit/85026f0c769e4000148a67d45a9e9b9c5108836f)
 * @author schteppe / https://github.com/schteppe
 * @see https://www.altdevblogaday.com/2011/05/13/contact-generation-between-3d-convex-meshes/
 *
 * @todo Move the clipping functions to ContactGenerator?
 * @todo Automatically merge coplanar polygons in constructor.
 * @example
 *     const convexShape = new CANNON.ConvexPolyhedron({ vertices, faces })
 *     const convexBody = new CANNON.Body({ mass: 1, shape: convexShape })
 *     world.addBody(convexBody)
 */
class ConvexPolyhedron extends Shape {
  /** vertices */

  /**
   * Array of integer arrays, indicating which vertices each face consists of
   */

  /** faceNormals */

  /** worldVertices */

  /** worldVerticesNeedsUpdate */

  /** worldFaceNormals */

  /** worldFaceNormalsNeedsUpdate */

  /**
   * If given, these locally defined, normalized axes are the only ones being checked when doing separating axis check.
   */

  /** uniqueEdges */

  /**
   * @param vertices An array of Vec3's
   * @param faces Array of integer arrays, describing which vertices that is included in each face.
   */
  constructor(props) {
    if (props === void 0) {
      props = {};
    }

    const {
      vertices = [],
      faces = [],
      normals = [],
      axes,
      boundingSphereRadius
    } = props;
    super({
      type: Shape.types.CONVEXPOLYHEDRON
    });
    this.vertices = vertices;
    this.faces = faces;
    this.faceNormals = normals;

    if (this.faceNormals.length === 0) {
      this.computeNormals();
    }

    if (!boundingSphereRadius) {
      this.updateBoundingSphereRadius();
    } else {
      this.boundingSphereRadius = boundingSphereRadius;
    }

    this.worldVertices = []; // World transformed version of .vertices

    this.worldVerticesNeedsUpdate = true;
    this.worldFaceNormals = []; // World transformed version of .faceNormals

    this.worldFaceNormalsNeedsUpdate = true;
    this.uniqueAxes = axes ? axes.slice() : null;
    this.uniqueEdges = [];
    this.computeEdges();
  }
  /**
   * Computes uniqueEdges
   */


  computeEdges() {
    const faces = this.faces;
    const vertices = this.vertices;
    const edges = this.uniqueEdges;
    edges.length = 0;
    const edge = new Vec3();

    for (let i = 0; i !== faces.length; i++) {
      const face = faces[i];
      const numVertices = face.length;

      for (let j = 0; j !== numVertices; j++) {
        const k = (j + 1) % numVertices;
        vertices[face[j]].vsub(vertices[face[k]], edge);
        edge.normalize();
        let found = false;

        for (let p = 0; p !== edges.length; p++) {
          if (edges[p].almostEquals(edge) || edges[p].almostEquals(edge)) {
            found = true;
            break;
          }
        }

        if (!found) {
          edges.push(edge.clone());
        }
      }
    }
  }
  /**
   * Compute the normals of the faces.
   * Will reuse existing Vec3 objects in the `faceNormals` array if they exist.
   */


  computeNormals() {
    this.faceNormals.length = this.faces.length; // Generate normals

    for (let i = 0; i < this.faces.length; i++) {
      // Check so all vertices exists for this face
      for (let j = 0; j < this.faces[i].length; j++) {
        if (!this.vertices[this.faces[i][j]]) {
          throw new Error(`Vertex ${this.faces[i][j]} not found!`);
        }
      }

      const n = this.faceNormals[i] || new Vec3();
      this.getFaceNormal(i, n);
      n.negate(n);
      this.faceNormals[i] = n;
      const vertex = this.vertices[this.faces[i][0]];

      if (n.dot(vertex) < 0) {
        console.error(`.faceNormals[${i}] = Vec3(${n.toString()}) looks like it points into the shape? The vertices follow. Make sure they are ordered CCW around the normal, using the right hand rule.`);

        for (let j = 0; j < this.faces[i].length; j++) {
          console.warn(`.vertices[${this.faces[i][j]}] = Vec3(${this.vertices[this.faces[i][j]].toString()})`);
        }
      }
    }
  }
  /**
   * Compute the normal of a face from its vertices
   */


  getFaceNormal(i, target) {
    const f = this.faces[i];
    const va = this.vertices[f[0]];
    const vb = this.vertices[f[1]];
    const vc = this.vertices[f[2]];
    ConvexPolyhedron.computeNormal(va, vb, vc, target);
  }
  /**
   * Get face normal given 3 vertices
   */


  static computeNormal(va, vb, vc, target) {
    const cb = new Vec3();
    const ab = new Vec3();
    vb.vsub(va, ab);
    vc.vsub(vb, cb);
    cb.cross(ab, target);

    if (!target.isZero()) {
      target.normalize();
    }
  }
  /**
   * @param minDist Clamp distance
   * @param result The an array of contact point objects, see clipFaceAgainstHull
   */


  clipAgainstHull(posA, quatA, hullB, posB, quatB, separatingNormal, minDist, maxDist, result) {
    const WorldNormal = new Vec3();
    let closestFaceB = -1;
    let dmax = -Number.MAX_VALUE;

    for (let face = 0; face < hullB.faces.length; face++) {
      WorldNormal.copy(hullB.faceNormals[face]);
      quatB.vmult(WorldNormal, WorldNormal);
      const d = WorldNormal.dot(separatingNormal);

      if (d > dmax) {
        dmax = d;
        closestFaceB = face;
      }
    }

    const worldVertsB1 = [];

    for (let i = 0; i < hullB.faces[closestFaceB].length; i++) {
      const b = hullB.vertices[hullB.faces[closestFaceB][i]];
      const worldb = new Vec3();
      worldb.copy(b);
      quatB.vmult(worldb, worldb);
      posB.vadd(worldb, worldb);
      worldVertsB1.push(worldb);
    }

    if (closestFaceB >= 0) {
      this.clipFaceAgainstHull(separatingNormal, posA, quatA, worldVertsB1, minDist, maxDist, result);
    }
  }
  /**
   * Find the separating axis between this hull and another
   * @param target The target vector to save the axis in
   * @return Returns false if a separation is found, else true
   */


  findSeparatingAxis(hullB, posA, quatA, posB, quatB, target, faceListA, faceListB) {
    const faceANormalWS3 = new Vec3();
    const Worldnormal1 = new Vec3();
    const deltaC = new Vec3();
    const worldEdge0 = new Vec3();
    const worldEdge1 = new Vec3();
    const Cross = new Vec3();
    let dmin = Number.MAX_VALUE;
    const hullA = this;

    if (!hullA.uniqueAxes) {
      const numFacesA = faceListA ? faceListA.length : hullA.faces.length; // Test face normals from hullA

      for (let i = 0; i < numFacesA; i++) {
        const fi = faceListA ? faceListA[i] : i; // Get world face normal

        faceANormalWS3.copy(hullA.faceNormals[fi]);
        quatA.vmult(faceANormalWS3, faceANormalWS3);
        const d = hullA.testSepAxis(faceANormalWS3, hullB, posA, quatA, posB, quatB);

        if (d === false) {
          return false;
        }

        if (d < dmin) {
          dmin = d;
          target.copy(faceANormalWS3);
        }
      }
    } else {
      // Test unique axes
      for (let i = 0; i !== hullA.uniqueAxes.length; i++) {
        // Get world axis
        quatA.vmult(hullA.uniqueAxes[i], faceANormalWS3);
        const d = hullA.testSepAxis(faceANormalWS3, hullB, posA, quatA, posB, quatB);

        if (d === false) {
          return false;
        }

        if (d < dmin) {
          dmin = d;
          target.copy(faceANormalWS3);
        }
      }
    }

    if (!hullB.uniqueAxes) {
      // Test face normals from hullB
      const numFacesB = faceListB ? faceListB.length : hullB.faces.length;

      for (let i = 0; i < numFacesB; i++) {
        const fi = faceListB ? faceListB[i] : i;
        Worldnormal1.copy(hullB.faceNormals[fi]);
        quatB.vmult(Worldnormal1, Worldnormal1);
        const d = hullA.testSepAxis(Worldnormal1, hullB, posA, quatA, posB, quatB);

        if (d === false) {
          return false;
        }

        if (d < dmin) {
          dmin = d;
          target.copy(Worldnormal1);
        }
      }
    } else {
      // Test unique axes in B
      for (let i = 0; i !== hullB.uniqueAxes.length; i++) {
        quatB.vmult(hullB.uniqueAxes[i], Worldnormal1);
        const d = hullA.testSepAxis(Worldnormal1, hullB, posA, quatA, posB, quatB);

        if (d === false) {
          return false;
        }

        if (d < dmin) {
          dmin = d;
          target.copy(Worldnormal1);
        }
      }
    } // Test edges


    for (let e0 = 0; e0 !== hullA.uniqueEdges.length; e0++) {
      // Get world edge
      quatA.vmult(hullA.uniqueEdges[e0], worldEdge0);

      for (let e1 = 0; e1 !== hullB.uniqueEdges.length; e1++) {
        // Get world edge 2
        quatB.vmult(hullB.uniqueEdges[e1], worldEdge1);
        worldEdge0.cross(worldEdge1, Cross);

        if (!Cross.almostZero()) {
          Cross.normalize();
          const dist = hullA.testSepAxis(Cross, hullB, posA, quatA, posB, quatB);

          if (dist === false) {
            return false;
          }

          if (dist < dmin) {
            dmin = dist;
            target.copy(Cross);
          }
        }
      }
    }

    posB.vsub(posA, deltaC);

    if (deltaC.dot(target) > 0.0) {
      target.negate(target);
    }

    return true;
  }
  /**
   * Test separating axis against two hulls. Both hulls are projected onto the axis and the overlap size is returned if there is one.
   * @return The overlap depth, or FALSE if no penetration.
   */


  testSepAxis(axis, hullB, posA, quatA, posB, quatB) {
    const hullA = this;
    ConvexPolyhedron.project(hullA, axis, posA, quatA, maxminA);
    ConvexPolyhedron.project(hullB, axis, posB, quatB, maxminB);
    const maxA = maxminA[0];
    const minA = maxminA[1];
    const maxB = maxminB[0];
    const minB = maxminB[1];

    if (maxA < minB || maxB < minA) {
      return false; // Separated
    }

    const d0 = maxA - minB;
    const d1 = maxB - minA;
    const depth = d0 < d1 ? d0 : d1;
    return depth;
  }
  /**
   * calculateLocalInertia
   */


  calculateLocalInertia(mass, target) {
    // Approximate with box inertia
    // Exact inertia calculation is overkill, but see http://geometrictools.com/Documentation/PolyhedralMassProperties.pdf for the correct way to do it
    const aabbmax = new Vec3();
    const aabbmin = new Vec3();
    this.computeLocalAABB(aabbmin, aabbmax);
    const x = aabbmax.x - aabbmin.x;
    const y = aabbmax.y - aabbmin.y;
    const z = aabbmax.z - aabbmin.z;
    target.x = 1.0 / 12.0 * mass * (2 * y * 2 * y + 2 * z * 2 * z);
    target.y = 1.0 / 12.0 * mass * (2 * x * 2 * x + 2 * z * 2 * z);
    target.z = 1.0 / 12.0 * mass * (2 * y * 2 * y + 2 * x * 2 * x);
  }
  /**
   * @param face_i Index of the face
   */


  getPlaneConstantOfFace(face_i) {
    const f = this.faces[face_i];
    const n = this.faceNormals[face_i];
    const v = this.vertices[f[0]];
    const c = -n.dot(v);
    return c;
  }
  /**
   * Clip a face against a hull.
   * @param worldVertsB1 An array of Vec3 with vertices in the world frame.
   * @param minDist Distance clamping
   * @param Array result Array to store resulting contact points in. Will be objects with properties: point, depth, normal. These are represented in world coordinates.
   */


  clipFaceAgainstHull(separatingNormal, posA, quatA, worldVertsB1, minDist, maxDist, result) {
    const faceANormalWS = new Vec3();
    const edge0 = new Vec3();
    const WorldEdge0 = new Vec3();
    const worldPlaneAnormal1 = new Vec3();
    const planeNormalWS1 = new Vec3();
    const worldA1 = new Vec3();
    const localPlaneNormal = new Vec3();
    const planeNormalWS = new Vec3();
    const hullA = this;
    const worldVertsB2 = [];
    const pVtxIn = worldVertsB1;
    const pVtxOut = worldVertsB2;
    let closestFaceA = -1;
    let dmin = Number.MAX_VALUE; // Find the face with normal closest to the separating axis

    for (let face = 0; face < hullA.faces.length; face++) {
      faceANormalWS.copy(hullA.faceNormals[face]);
      quatA.vmult(faceANormalWS, faceANormalWS);
      const d = faceANormalWS.dot(separatingNormal);

      if (d < dmin) {
        dmin = d;
        closestFaceA = face;
      }
    }

    if (closestFaceA < 0) {
      return;
    } // Get the face and construct connected faces


    const polyA = hullA.faces[closestFaceA];
    polyA.connectedFaces = [];

    for (let i = 0; i < hullA.faces.length; i++) {
      for (let j = 0; j < hullA.faces[i].length; j++) {
        if (
        /* Sharing a vertex*/
        polyA.indexOf(hullA.faces[i][j]) !== -1 &&
        /* Not the one we are looking for connections from */
        i !== closestFaceA &&
        /* Not already added */
        polyA.connectedFaces.indexOf(i) === -1) {
          polyA.connectedFaces.push(i);
        }
      }
    } // Clip the polygon to the back of the planes of all faces of hull A,
    // that are adjacent to the witness face


    const numVerticesA = polyA.length;

    for (let i = 0; i < numVerticesA; i++) {
      const a = hullA.vertices[polyA[i]];
      const b = hullA.vertices[polyA[(i + 1) % numVerticesA]];
      a.vsub(b, edge0);
      WorldEdge0.copy(edge0);
      quatA.vmult(WorldEdge0, WorldEdge0);
      posA.vadd(WorldEdge0, WorldEdge0);
      worldPlaneAnormal1.copy(this.faceNormals[closestFaceA]);
      quatA.vmult(worldPlaneAnormal1, worldPlaneAnormal1);
      posA.vadd(worldPlaneAnormal1, worldPlaneAnormal1);
      WorldEdge0.cross(worldPlaneAnormal1, planeNormalWS1);
      planeNormalWS1.negate(planeNormalWS1);
      worldA1.copy(a);
      quatA.vmult(worldA1, worldA1);
      posA.vadd(worldA1, worldA1);
      const otherFace = polyA.connectedFaces[i];
      localPlaneNormal.copy(this.faceNormals[otherFace]);
      const localPlaneEq = this.getPlaneConstantOfFace(otherFace);
      planeNormalWS.copy(localPlaneNormal);
      quatA.vmult(planeNormalWS, planeNormalWS);
      const planeEqWS = localPlaneEq - planeNormalWS.dot(posA); // Clip face against our constructed plane

      this.clipFaceAgainstPlane(pVtxIn, pVtxOut, planeNormalWS, planeEqWS); // Throw away all clipped points, but save the remaining until next clip

      while (pVtxIn.length) {
        pVtxIn.shift();
      }

      while (pVtxOut.length) {
        pVtxIn.push(pVtxOut.shift());
      }
    } // only keep contact points that are behind the witness face


    localPlaneNormal.copy(this.faceNormals[closestFaceA]);
    const localPlaneEq = this.getPlaneConstantOfFace(closestFaceA);
    planeNormalWS.copy(localPlaneNormal);
    quatA.vmult(planeNormalWS, planeNormalWS);
    const planeEqWS = localPlaneEq - planeNormalWS.dot(posA);

    for (let i = 0; i < pVtxIn.length; i++) {
      let depth = planeNormalWS.dot(pVtxIn[i]) + planeEqWS; // ???

      if (depth <= minDist) {
        console.log(`clamped: depth=${depth} to minDist=${minDist}`);
        depth = minDist;
      }

      if (depth <= maxDist) {
        const point = pVtxIn[i];

        if (depth <= 1e-6) {
          const p = {
            point,
            normal: planeNormalWS,
            depth
          };
          result.push(p);
        }
      }
    }
  }
  /**
   * Clip a face in a hull against the back of a plane.
   * @param planeConstant The constant in the mathematical plane equation
   */


  clipFaceAgainstPlane(inVertices, outVertices, planeNormal, planeConstant) {
    let n_dot_first;
    let n_dot_last;
    const numVerts = inVertices.length;

    if (numVerts < 2) {
      return outVertices;
    }

    let firstVertex = inVertices[inVertices.length - 1];
    let lastVertex = inVertices[0];
    n_dot_first = planeNormal.dot(firstVertex) + planeConstant;

    for (let vi = 0; vi < numVerts; vi++) {
      lastVertex = inVertices[vi];
      n_dot_last = planeNormal.dot(lastVertex) + planeConstant;

      if (n_dot_first < 0) {
        if (n_dot_last < 0) {
          // Start < 0, end < 0, so output lastVertex
          const newv = new Vec3();
          newv.copy(lastVertex);
          outVertices.push(newv);
        } else {
          // Start < 0, end >= 0, so output intersection
          const newv = new Vec3();
          firstVertex.lerp(lastVertex, n_dot_first / (n_dot_first - n_dot_last), newv);
          outVertices.push(newv);
        }
      } else {
        if (n_dot_last < 0) {
          // Start >= 0, end < 0 so output intersection and end
          const newv = new Vec3();
          firstVertex.lerp(lastVertex, n_dot_first / (n_dot_first - n_dot_last), newv);
          outVertices.push(newv);
          outVertices.push(lastVertex);
        }
      }

      firstVertex = lastVertex;
      n_dot_first = n_dot_last;
    }

    return outVertices;
  }
  /**
   * Updates `.worldVertices` and sets `.worldVerticesNeedsUpdate` to false.
   */


  computeWorldVertices(position, quat) {
    while (this.worldVertices.length < this.vertices.length) {
      this.worldVertices.push(new Vec3());
    }

    const verts = this.vertices;
    const worldVerts = this.worldVertices;

    for (let i = 0; i !== this.vertices.length; i++) {
      quat.vmult(verts[i], worldVerts[i]);
      position.vadd(worldVerts[i], worldVerts[i]);
    }

    this.worldVerticesNeedsUpdate = false;
  }

  computeLocalAABB(aabbmin, aabbmax) {
    const vertices = this.vertices;
    aabbmin.set(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
    aabbmax.set(-Number.MAX_VALUE, -Number.MAX_VALUE, -Number.MAX_VALUE);

    for (let i = 0; i < this.vertices.length; i++) {
      const v = vertices[i];

      if (v.x < aabbmin.x) {
        aabbmin.x = v.x;
      } else if (v.x > aabbmax.x) {
        aabbmax.x = v.x;
      }

      if (v.y < aabbmin.y) {
        aabbmin.y = v.y;
      } else if (v.y > aabbmax.y) {
        aabbmax.y = v.y;
      }

      if (v.z < aabbmin.z) {
        aabbmin.z = v.z;
      } else if (v.z > aabbmax.z) {
        aabbmax.z = v.z;
      }
    }
  }
  /**
   * Updates `worldVertices` and sets `worldVerticesNeedsUpdate` to false.
   */


  computeWorldFaceNormals(quat) {
    const N = this.faceNormals.length;

    while (this.worldFaceNormals.length < N) {
      this.worldFaceNormals.push(new Vec3());
    }

    const normals = this.faceNormals;
    const worldNormals = this.worldFaceNormals;

    for (let i = 0; i !== N; i++) {
      quat.vmult(normals[i], worldNormals[i]);
    }

    this.worldFaceNormalsNeedsUpdate = false;
  }
  /**
   * updateBoundingSphereRadius
   */


  updateBoundingSphereRadius() {
    // Assume points are distributed with local (0,0,0) as center
    let max2 = 0;
    const verts = this.vertices;

    for (let i = 0; i !== verts.length; i++) {
      const norm2 = verts[i].lengthSquared();

      if (norm2 > max2) {
        max2 = norm2;
      }
    }

    this.boundingSphereRadius = Math.sqrt(max2);
  }
  /**
   * calculateWorldAABB
   */


  calculateWorldAABB(pos, quat, min, max) {
    const verts = this.vertices;
    let minx;
    let miny;
    let minz;
    let maxx;
    let maxy;
    let maxz;
    let tempWorldVertex = new Vec3();

    for (let i = 0; i < verts.length; i++) {
      tempWorldVertex.copy(verts[i]);
      quat.vmult(tempWorldVertex, tempWorldVertex);
      pos.vadd(tempWorldVertex, tempWorldVertex);
      const v = tempWorldVertex;

      if (minx === undefined || v.x < minx) {
        minx = v.x;
      }

      if (maxx === undefined || v.x > maxx) {
        maxx = v.x;
      }

      if (miny === undefined || v.y < miny) {
        miny = v.y;
      }

      if (maxy === undefined || v.y > maxy) {
        maxy = v.y;
      }

      if (minz === undefined || v.z < minz) {
        minz = v.z;
      }

      if (maxz === undefined || v.z > maxz) {
        maxz = v.z;
      }
    }

    min.set(minx, miny, minz);
    max.set(maxx, maxy, maxz);
  }
  /**
   * Get approximate convex volume
   */


  volume() {
    return 4.0 * Math.PI * this.boundingSphereRadius / 3.0;
  }
  /**
   * Get an average of all the vertices positions
   */


  getAveragePointLocal(target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const verts = this.vertices;

    for (let i = 0; i < verts.length; i++) {
      target.vadd(verts[i], target);
    }

    target.scale(1 / verts.length, target);
    return target;
  }
  /**
   * Transform all local points. Will change the .vertices
   */


  transformAllPoints(offset, quat) {
    const n = this.vertices.length;
    const verts = this.vertices; // Apply rotation

    if (quat) {
      // Rotate vertices
      for (let i = 0; i < n; i++) {
        const v = verts[i];
        quat.vmult(v, v);
      } // Rotate face normals


      for (let i = 0; i < this.faceNormals.length; i++) {
        const v = this.faceNormals[i];
        quat.vmult(v, v);
      }
      /*
            // Rotate edges
            for(let i=0; i<this.uniqueEdges.length; i++){
                const v = this.uniqueEdges[i];
                quat.vmult(v,v);
            }*/

    } // Apply offset


    if (offset) {
      for (let i = 0; i < n; i++) {
        const v = verts[i];
        v.vadd(offset, v);
      }
    }
  }
  /**
   * Checks whether p is inside the polyhedra. Must be in local coords.
   * The point lies outside of the convex hull of the other points if and only if the direction
   * of all the vectors from it to those other points are on less than one half of a sphere around it.
   * @param p A point given in local coordinates
   */


  pointIsInside(p) {
    const verts = this.vertices;
    const faces = this.faces;
    const normals = this.faceNormals;
    const pointInside = new Vec3();
    this.getAveragePointLocal(pointInside);

    for (let i = 0; i < this.faces.length; i++) {
      let n = normals[i];
      const v = verts[faces[i][0]]; // We only need one point in the face
      // This dot product determines which side of the edge the point is

      const vToP = new Vec3();
      p.vsub(v, vToP);
      const r1 = n.dot(vToP);
      const vToPointInside = new Vec3();
      pointInside.vsub(v, vToPointInside);
      const r2 = n.dot(vToPointInside);

      if (r1 < 0 && r2 > 0 || r1 > 0 && r2 < 0) {
        return false; // Encountered some other sign. Exit.
      }
    } // If we got here, all dot products were of the same sign.


    return -1;
  }
  /**
   * Get max and min dot product of a convex hull at position (pos,quat) projected onto an axis.
   * Results are saved in the array maxmin.
   * @param result result[0] and result[1] will be set to maximum and minimum, respectively.
   */


  static project(shape, axis, pos, quat, result) {
    const n = shape.vertices.length;
    const localAxis = project_localAxis;
    let max = 0;
    let min = 0;
    const localOrigin = project_localOrigin;
    const vs = shape.vertices;
    localOrigin.setZero(); // Transform the axis to local

    Transform.vectorToLocalFrame(pos, quat, axis, localAxis);
    Transform.pointToLocalFrame(pos, quat, localOrigin, localOrigin);
    const add = localOrigin.dot(localAxis);
    min = max = vs[0].dot(localAxis);

    for (let i = 1; i < n; i++) {
      const val = vs[i].dot(localAxis);

      if (val > max) {
        max = val;
      }

      if (val < min) {
        min = val;
      }
    }

    min -= add;
    max -= add;

    if (min > max) {
      // Inconsistent - swap
      const temp = min;
      min = max;
      max = temp;
    } // Output


    result[0] = max;
    result[1] = min;
  }

}
const maxminA = [];
const maxminB = [];
new Vec3();
const project_localAxis = new Vec3();
const project_localOrigin = new Vec3();

/**
 * A 3d box shape.
 * @example
 *     const size = 1
 *     const halfExtents = new CANNON.Vec3(size, size, size)
 *     const boxShape = new CANNON.Box(halfExtents)
 *     const boxBody = new CANNON.Body({ mass: 1, shape: boxShape })
 *     world.addBody(boxBody)
 */
class Box extends Shape {
  /**
   * The half extents of the box.
   */

  /**
   * Used by the contact generator to make contacts with other convex polyhedra for example.
   */
  constructor(halfExtents) {
    super({
      type: Shape.types.BOX
    });
    this.halfExtents = halfExtents;
    this.convexPolyhedronRepresentation = null;
    this.updateConvexPolyhedronRepresentation();
    this.updateBoundingSphereRadius();
  }
  /**
   * Updates the local convex polyhedron representation used for some collisions.
   */


  updateConvexPolyhedronRepresentation() {
    const sx = this.halfExtents.x;
    const sy = this.halfExtents.y;
    const sz = this.halfExtents.z;
    const V = Vec3;
    const vertices = [new V(-sx, -sy, -sz), new V(sx, -sy, -sz), new V(sx, sy, -sz), new V(-sx, sy, -sz), new V(-sx, -sy, sz), new V(sx, -sy, sz), new V(sx, sy, sz), new V(-sx, sy, sz)];
    const faces = [[3, 2, 1, 0], // -z
    [4, 5, 6, 7], // +z
    [5, 4, 0, 1], // -y
    [2, 3, 7, 6], // +y
    [0, 4, 7, 3], // -x
    [1, 2, 6, 5] // +x
    ];
    const axes = [new V(0, 0, 1), new V(0, 1, 0), new V(1, 0, 0)];
    const h = new ConvexPolyhedron({
      vertices,
      faces,
      axes
    });
    this.convexPolyhedronRepresentation = h;
    h.material = this.material;
  }
  /**
   * Calculate the inertia of the box.
   */


  calculateLocalInertia(mass, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    Box.calculateInertia(this.halfExtents, mass, target);
    return target;
  }

  static calculateInertia(halfExtents, mass, target) {
    const e = halfExtents;
    target.x = 1.0 / 12.0 * mass * (2 * e.y * 2 * e.y + 2 * e.z * 2 * e.z);
    target.y = 1.0 / 12.0 * mass * (2 * e.x * 2 * e.x + 2 * e.z * 2 * e.z);
    target.z = 1.0 / 12.0 * mass * (2 * e.y * 2 * e.y + 2 * e.x * 2 * e.x);
  }
  /**
   * Get the box 6 side normals
   * @param sixTargetVectors An array of 6 vectors, to store the resulting side normals in.
   * @param quat Orientation to apply to the normal vectors. If not provided, the vectors will be in respect to the local frame.
   */


  getSideNormals(sixTargetVectors, quat) {
    const sides = sixTargetVectors;
    const ex = this.halfExtents;
    sides[0].set(ex.x, 0, 0);
    sides[1].set(0, ex.y, 0);
    sides[2].set(0, 0, ex.z);
    sides[3].set(-ex.x, 0, 0);
    sides[4].set(0, -ex.y, 0);
    sides[5].set(0, 0, -ex.z);

    if (quat !== undefined) {
      for (let i = 0; i !== sides.length; i++) {
        quat.vmult(sides[i], sides[i]);
      }
    }

    return sides;
  }
  /**
   * Returns the volume of the box.
   */


  volume() {
    return 8.0 * this.halfExtents.x * this.halfExtents.y * this.halfExtents.z;
  }
  /**
   * updateBoundingSphereRadius
   */


  updateBoundingSphereRadius() {
    this.boundingSphereRadius = this.halfExtents.length();
  }
  /**
   * forEachWorldCorner
   */


  forEachWorldCorner(pos, quat, callback) {
    const e = this.halfExtents;
    const corners = [[e.x, e.y, e.z], [-e.x, e.y, e.z], [-e.x, -e.y, e.z], [-e.x, -e.y, -e.z], [e.x, -e.y, -e.z], [e.x, e.y, -e.z], [-e.x, e.y, -e.z], [e.x, -e.y, e.z]];

    for (let i = 0; i < corners.length; i++) {
      worldCornerTempPos.set(corners[i][0], corners[i][1], corners[i][2]);
      quat.vmult(worldCornerTempPos, worldCornerTempPos);
      pos.vadd(worldCornerTempPos, worldCornerTempPos);
      callback(worldCornerTempPos.x, worldCornerTempPos.y, worldCornerTempPos.z);
    }
  }
  /**
   * calculateWorldAABB
   */


  calculateWorldAABB(pos, quat, min, max) {
    const e = this.halfExtents;
    worldCornersTemp[0].set(e.x, e.y, e.z);
    worldCornersTemp[1].set(-e.x, e.y, e.z);
    worldCornersTemp[2].set(-e.x, -e.y, e.z);
    worldCornersTemp[3].set(-e.x, -e.y, -e.z);
    worldCornersTemp[4].set(e.x, -e.y, -e.z);
    worldCornersTemp[5].set(e.x, e.y, -e.z);
    worldCornersTemp[6].set(-e.x, e.y, -e.z);
    worldCornersTemp[7].set(e.x, -e.y, e.z);
    const wc = worldCornersTemp[0];
    quat.vmult(wc, wc);
    pos.vadd(wc, wc);
    max.copy(wc);
    min.copy(wc);

    for (let i = 1; i < 8; i++) {
      const wc = worldCornersTemp[i];
      quat.vmult(wc, wc);
      pos.vadd(wc, wc);
      const x = wc.x;
      const y = wc.y;
      const z = wc.z;

      if (x > max.x) {
        max.x = x;
      }

      if (y > max.y) {
        max.y = y;
      }

      if (z > max.z) {
        max.z = z;
      }

      if (x < min.x) {
        min.x = x;
      }

      if (y < min.y) {
        min.y = y;
      }

      if (z < min.z) {
        min.z = z;
      }
    } // Get each axis max
    // min.set(Infinity,Infinity,Infinity);
    // max.set(-Infinity,-Infinity,-Infinity);
    // this.forEachWorldCorner(pos,quat,function(x,y,z){
    //     if(x > max.x){
    //         max.x = x;
    //     }
    //     if(y > max.y){
    //         max.y = y;
    //     }
    //     if(z > max.z){
    //         max.z = z;
    //     }
    //     if(x < min.x){
    //         min.x = x;
    //     }
    //     if(y < min.y){
    //         min.y = y;
    //     }
    //     if(z < min.z){
    //         min.z = z;
    //     }
    // });

  }

}
const worldCornerTempPos = new Vec3();
const worldCornersTemp = [new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3()];

/**
 * BODY_TYPES
 */
const BODY_TYPES = {
  /** DYNAMIC */
  DYNAMIC: 1,

  /** STATIC */
  STATIC: 2,

  /** KINEMATIC */
  KINEMATIC: 4
};
/**
 * BodyType
 */

/**
 * BODY_SLEEP_STATES
 */
const BODY_SLEEP_STATES = {
  /** AWAKE */
  AWAKE: 0,

  /** SLEEPY */
  SLEEPY: 1,

  /** SLEEPING */
  SLEEPING: 2
};
/**
 * BodySleepState
 */

/**
 * Base class for all body types.
 * @example
 *     const shape = new CANNON.Sphere(1)
 *     const body = new CANNON.Body({
 *       mass: 1,
 *       shape,
 *     })
 *     world.addBody(body)
 */
class Body extends EventTarget {
  /**
   * Dispatched after two bodies collide. This event is dispatched on each
   * of the two bodies involved in the collision.
   * @event collide
   * @param body The body that was involved in the collision.
   * @param contact The details of the collision.
   */

  /**
   * A dynamic body is fully simulated. Can be moved manually by the user, but normally they move according to forces. A dynamic body can collide with all body types. A dynamic body always has finite, non-zero mass.
   */

  /**
   * A static body does not move during simulation and behaves as if it has infinite mass. Static bodies can be moved manually by setting the position of the body. The velocity of a static body is always zero. Static bodies do not collide with other static or kinematic bodies.
   */

  /**
   * A kinematic body moves under simulation according to its velocity. They do not respond to forces. They can be moved manually, but normally a kinematic body is moved by setting its velocity. A kinematic body behaves as if it has infinite mass. Kinematic bodies do not collide with other static or kinematic bodies.
   */

  /**
   * AWAKE
   */

  /**
   * SLEEPY
   */

  /**
   * SLEEPING
   */

  /**
   * Dispatched after a sleeping body has woken up.
   * @event wakeup
   */

  /**
   * Dispatched after a body has gone in to the sleepy state.
   * @event sleepy
   */

  /**
   * Dispatched after a body has fallen asleep.
   * @event sleep
   */
  constructor(options) {
    if (options === void 0) {
      options = {};
    }

    super();
    this.id = Body.idCounter++;
    this.index = -1;
    this.world = null;
    this.vlambda = new Vec3();
    this.collisionFilterGroup = typeof options.collisionFilterGroup === 'number' ? options.collisionFilterGroup : 1;
    this.collisionFilterMask = typeof options.collisionFilterMask === 'number' ? options.collisionFilterMask : -1;
    this.collisionResponse = typeof options.collisionResponse === 'boolean' ? options.collisionResponse : true;
    this.position = new Vec3();
    this.previousPosition = new Vec3();
    this.interpolatedPosition = new Vec3();
    this.initPosition = new Vec3();

    if (options.position) {
      this.position.copy(options.position);
      this.previousPosition.copy(options.position);
      this.interpolatedPosition.copy(options.position);
      this.initPosition.copy(options.position);
    }

    this.velocity = new Vec3();

    if (options.velocity) {
      this.velocity.copy(options.velocity);
    }

    this.initVelocity = new Vec3();
    this.force = new Vec3();
    const mass = typeof options.mass === 'number' ? options.mass : 0;
    this.mass = mass;
    this.invMass = mass > 0 ? 1.0 / mass : 0;
    this.material = options.material || null;
    this.linearDamping = typeof options.linearDamping === 'number' ? options.linearDamping : 0.01;
    this.type = mass <= 0.0 ? Body.STATIC : Body.DYNAMIC;

    if (typeof options.type === typeof Body.STATIC) {
      this.type = options.type;
    }

    this.allowSleep = typeof options.allowSleep !== 'undefined' ? options.allowSleep : true;
    this.sleepState = Body.AWAKE;
    this.sleepSpeedLimit = typeof options.sleepSpeedLimit !== 'undefined' ? options.sleepSpeedLimit : 0.1;
    this.sleepTimeLimit = typeof options.sleepTimeLimit !== 'undefined' ? options.sleepTimeLimit : 1;
    this.timeLastSleepy = 0;
    this.wakeUpAfterNarrowphase = false;
    this.torque = new Vec3();
    this.quaternion = new Quaternion();
    this.initQuaternion = new Quaternion();
    this.previousQuaternion = new Quaternion();
    this.interpolatedQuaternion = new Quaternion();

    if (options.quaternion) {
      this.quaternion.copy(options.quaternion);
      this.initQuaternion.copy(options.quaternion);
      this.previousQuaternion.copy(options.quaternion);
      this.interpolatedQuaternion.copy(options.quaternion);
    }

    this.angularVelocity = new Vec3();

    if (options.angularVelocity) {
      this.angularVelocity.copy(options.angularVelocity);
    }

    this.initAngularVelocity = new Vec3();
    this.shapes = [];
    this.shapeOffsets = [];
    this.shapeOrientations = [];
    this.inertia = new Vec3();
    this.invInertia = new Vec3();
    this.invInertiaWorld = new Mat3();
    this.invMassSolve = 0;
    this.invInertiaSolve = new Vec3();
    this.invInertiaWorldSolve = new Mat3();
    this.fixedRotation = typeof options.fixedRotation !== 'undefined' ? options.fixedRotation : false;
    this.angularDamping = typeof options.angularDamping !== 'undefined' ? options.angularDamping : 0.01;
    this.linearFactor = new Vec3(1, 1, 1);

    if (options.linearFactor) {
      this.linearFactor.copy(options.linearFactor);
    }

    this.angularFactor = new Vec3(1, 1, 1);

    if (options.angularFactor) {
      this.angularFactor.copy(options.angularFactor);
    }

    this.aabb = new AABB();
    this.aabbNeedsUpdate = true;
    this.boundingRadius = 0;
    this.wlambda = new Vec3();
    this.isTrigger = Boolean(options.isTrigger);

    if (options.shape) {
      this.addShape(options.shape);
    }

    this.updateMassProperties();
  }
  /**
   * Wake the body up.
   */


  wakeUp() {
    const prevState = this.sleepState;
    this.sleepState = Body.AWAKE;
    this.wakeUpAfterNarrowphase = false;

    if (prevState === Body.SLEEPING) {
      this.dispatchEvent(Body.wakeupEvent);
    }
  }
  /**
   * Force body sleep
   */


  sleep() {
    this.sleepState = Body.SLEEPING;
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);
    this.wakeUpAfterNarrowphase = false;
  }
  /**
   * Called every timestep to update internal sleep timer and change sleep state if needed.
   * @param time The world time in seconds
   */


  sleepTick(time) {
    if (this.allowSleep) {
      const sleepState = this.sleepState;
      const speedSquared = this.velocity.lengthSquared() + this.angularVelocity.lengthSquared();
      const speedLimitSquared = this.sleepSpeedLimit ** 2;

      if (sleepState === Body.AWAKE && speedSquared < speedLimitSquared) {
        this.sleepState = Body.SLEEPY; // Sleepy

        this.timeLastSleepy = time;
        this.dispatchEvent(Body.sleepyEvent);
      } else if (sleepState === Body.SLEEPY && speedSquared > speedLimitSquared) {
        this.wakeUp(); // Wake up
      } else if (sleepState === Body.SLEEPY && time - this.timeLastSleepy > this.sleepTimeLimit) {
        this.sleep(); // Sleeping

        this.dispatchEvent(Body.sleepEvent);
      }
    }
  }
  /**
   * If the body is sleeping, it should be immovable / have infinite mass during solve. We solve it by having a separate "solve mass".
   */


  updateSolveMassProperties() {
    if (this.sleepState === Body.SLEEPING || this.type === Body.KINEMATIC) {
      this.invMassSolve = 0;
      this.invInertiaSolve.setZero();
      this.invInertiaWorldSolve.setZero();
    } else {
      this.invMassSolve = this.invMass;
      this.invInertiaSolve.copy(this.invInertia);
      this.invInertiaWorldSolve.copy(this.invInertiaWorld);
    }
  }
  /**
   * Convert a world point to local body frame.
   */


  pointToLocalFrame(worldPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    worldPoint.vsub(this.position, result);
    this.quaternion.conjugate().vmult(result, result);
    return result;
  }
  /**
   * Convert a world vector to local body frame.
   */


  vectorToLocalFrame(worldVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    this.quaternion.conjugate().vmult(worldVector, result);
    return result;
  }
  /**
   * Convert a local body point to world frame.
   */


  pointToWorldFrame(localPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    this.quaternion.vmult(localPoint, result);
    result.vadd(this.position, result);
    return result;
  }
  /**
   * Convert a local body point to world frame.
   */


  vectorToWorldFrame(localVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }

    this.quaternion.vmult(localVector, result);
    return result;
  }
  /**
   * Add a shape to the body with a local offset and orientation.
   * @return The body object, for chainability.
   */


  addShape(shape, _offset, _orientation) {
    const offset = new Vec3();
    const orientation = new Quaternion();

    if (_offset) {
      offset.copy(_offset);
    }

    if (_orientation) {
      orientation.copy(_orientation);
    }

    this.shapes.push(shape);
    this.shapeOffsets.push(offset);
    this.shapeOrientations.push(orientation);
    this.updateMassProperties();
    this.updateBoundingRadius();
    this.aabbNeedsUpdate = true;
    shape.body = this;
    return this;
  }
  /**
   * Remove a shape from the body.
   * @return The body object, for chainability.
   */


  removeShape(shape) {
    const index = this.shapes.indexOf(shape);

    if (index === -1) {
      console.warn('Shape does not belong to the body');
      return this;
    }

    this.shapes.splice(index, 1);
    this.shapeOffsets.splice(index, 1);
    this.shapeOrientations.splice(index, 1);
    this.updateMassProperties();
    this.updateBoundingRadius();
    this.aabbNeedsUpdate = true;
    shape.body = null;
    return this;
  }
  /**
   * Update the bounding radius of the body. Should be done if any of the shapes are changed.
   */


  updateBoundingRadius() {
    const shapes = this.shapes;
    const shapeOffsets = this.shapeOffsets;
    const N = shapes.length;
    let radius = 0;

    for (let i = 0; i !== N; i++) {
      const shape = shapes[i];
      shape.updateBoundingSphereRadius();
      const offset = shapeOffsets[i].length();
      const r = shape.boundingSphereRadius;

      if (offset + r > radius) {
        radius = offset + r;
      }
    }

    this.boundingRadius = radius;
  }
  /**
   * Updates the .aabb
   */


  updateAABB() {
    const shapes = this.shapes;
    const shapeOffsets = this.shapeOffsets;
    const shapeOrientations = this.shapeOrientations;
    const N = shapes.length;
    const offset = tmpVec;
    const orientation = tmpQuat;
    const bodyQuat = this.quaternion;
    const aabb = this.aabb;
    const shapeAABB = updateAABB_shapeAABB;

    for (let i = 0; i !== N; i++) {
      const shape = shapes[i]; // Get shape world position

      bodyQuat.vmult(shapeOffsets[i], offset);
      offset.vadd(this.position, offset); // Get shape world quaternion

      bodyQuat.mult(shapeOrientations[i], orientation); // Get shape AABB

      shape.calculateWorldAABB(offset, orientation, shapeAABB.lowerBound, shapeAABB.upperBound);

      if (i === 0) {
        aabb.copy(shapeAABB);
      } else {
        aabb.extend(shapeAABB);
      }
    }

    this.aabbNeedsUpdate = false;
  }
  /**
   * Update `.inertiaWorld` and `.invInertiaWorld`
   */


  updateInertiaWorld(force) {
    const I = this.invInertia;

    if (I.x === I.y && I.y === I.z && !force) ; else {
      const m1 = uiw_m1;
      const m2 = uiw_m2;
      uiw_m3;
      m1.setRotationFromQuaternion(this.quaternion);
      m1.transpose(m2);
      m1.scale(I, m1);
      m1.mmult(m2, this.invInertiaWorld);
    }
  }
  /**
   * Apply force to a point of the body. This could for example be a point on the Body surface.
   * Applying force this way will add to Body.force and Body.torque.
   * @param force The amount of force to add.
   * @param relativePoint A point relative to the center of mass to apply the force on.
   */


  applyForce(force, relativePoint) {
    if (relativePoint === void 0) {
      relativePoint = new Vec3();
    }

    // Needed?
    if (this.type !== Body.DYNAMIC) {
      return;
    }

    if (this.sleepState === Body.SLEEPING) {
      this.wakeUp();
    } // Compute produced rotational force


    const rotForce = Body_applyForce_rotForce;
    relativePoint.cross(force, rotForce); // Add linear force

    this.force.vadd(force, this.force); // Add rotational force

    this.torque.vadd(rotForce, this.torque);
  }
  /**
   * Apply force to a local point in the body.
   * @param force The force vector to apply, defined locally in the body frame.
   * @param localPoint A local point in the body to apply the force on.
   */


  applyLocalForce(localForce, localPoint) {
    if (localPoint === void 0) {
      localPoint = new Vec3();
    }

    if (this.type !== Body.DYNAMIC) {
      return;
    }

    const worldForce = Body_applyLocalForce_worldForce;
    const relativePointWorld = Body_applyLocalForce_relativePointWorld; // Transform the force vector to world space

    this.vectorToWorldFrame(localForce, worldForce);
    this.vectorToWorldFrame(localPoint, relativePointWorld);
    this.applyForce(worldForce, relativePointWorld);
  }
  /**
   * Apply torque to the body.
   * @param torque The amount of torque to add.
   */


  applyTorque(torque) {
    if (this.type !== Body.DYNAMIC) {
      return;
    }

    if (this.sleepState === Body.SLEEPING) {
      this.wakeUp();
    } // Add rotational force


    this.torque.vadd(torque, this.torque);
  }
  /**
   * Apply impulse to a point of the body. This could for example be a point on the Body surface.
   * An impulse is a force added to a body during a short period of time (impulse = force * time).
   * Impulses will be added to Body.velocity and Body.angularVelocity.
   * @param impulse The amount of impulse to add.
   * @param relativePoint A point relative to the center of mass to apply the force on.
   */


  applyImpulse(impulse, relativePoint) {
    if (relativePoint === void 0) {
      relativePoint = new Vec3();
    }

    if (this.type !== Body.DYNAMIC) {
      return;
    }

    if (this.sleepState === Body.SLEEPING) {
      this.wakeUp();
    } // Compute point position relative to the body center


    const r = relativePoint; // Compute produced central impulse velocity

    const velo = Body_applyImpulse_velo;
    velo.copy(impulse);
    velo.scale(this.invMass, velo); // Add linear impulse

    this.velocity.vadd(velo, this.velocity); // Compute produced rotational impulse velocity

    const rotVelo = Body_applyImpulse_rotVelo;
    r.cross(impulse, rotVelo);
    /*
     rotVelo.x *= this.invInertia.x;
     rotVelo.y *= this.invInertia.y;
     rotVelo.z *= this.invInertia.z;
     */

    this.invInertiaWorld.vmult(rotVelo, rotVelo); // Add rotational Impulse

    this.angularVelocity.vadd(rotVelo, this.angularVelocity);
  }
  /**
   * Apply locally-defined impulse to a local point in the body.
   * @param force The force vector to apply, defined locally in the body frame.
   * @param localPoint A local point in the body to apply the force on.
   */


  applyLocalImpulse(localImpulse, localPoint) {
    if (localPoint === void 0) {
      localPoint = new Vec3();
    }

    if (this.type !== Body.DYNAMIC) {
      return;
    }

    const worldImpulse = Body_applyLocalImpulse_worldImpulse;
    const relativePointWorld = Body_applyLocalImpulse_relativePoint; // Transform the force vector to world space

    this.vectorToWorldFrame(localImpulse, worldImpulse);
    this.vectorToWorldFrame(localPoint, relativePointWorld);
    this.applyImpulse(worldImpulse, relativePointWorld);
  }
  /**
   * Should be called whenever you change the body shape or mass.
   */


  updateMassProperties() {
    const halfExtents = Body_updateMassProperties_halfExtents;
    this.invMass = this.mass > 0 ? 1.0 / this.mass : 0;
    const I = this.inertia;
    const fixed = this.fixedRotation; // Approximate with AABB box

    this.updateAABB();
    halfExtents.set((this.aabb.upperBound.x - this.aabb.lowerBound.x) / 2, (this.aabb.upperBound.y - this.aabb.lowerBound.y) / 2, (this.aabb.upperBound.z - this.aabb.lowerBound.z) / 2);
    Box.calculateInertia(halfExtents, this.mass, I);
    this.invInertia.set(I.x > 0 && !fixed ? 1.0 / I.x : 0, I.y > 0 && !fixed ? 1.0 / I.y : 0, I.z > 0 && !fixed ? 1.0 / I.z : 0);
    this.updateInertiaWorld(true);
  }
  /**
   * Get world velocity of a point in the body.
   * @param worldPoint
   * @param result
   * @return The result vector.
   */


  getVelocityAtWorldPoint(worldPoint, result) {
    const r = new Vec3();
    worldPoint.vsub(this.position, r);
    this.angularVelocity.cross(r, result);
    this.velocity.vadd(result, result);
    return result;
  }
  /**
   * Move the body forward in time.
   * @param dt Time step
   * @param quatNormalize Set to true to normalize the body quaternion
   * @param quatNormalizeFast If the quaternion should be normalized using "fast" quaternion normalization
   */


  integrate(dt, quatNormalize, quatNormalizeFast) {
    // Save previous position
    this.previousPosition.copy(this.position);
    this.previousQuaternion.copy(this.quaternion);

    if (!(this.type === Body.DYNAMIC || this.type === Body.KINEMATIC) || this.sleepState === Body.SLEEPING) {
      // Only for dynamic
      return;
    }

    const velo = this.velocity;
    const angularVelo = this.angularVelocity;
    const pos = this.position;
    const force = this.force;
    const torque = this.torque;
    const quat = this.quaternion;
    const invMass = this.invMass;
    const invInertia = this.invInertiaWorld;
    const linearFactor = this.linearFactor;
    const iMdt = invMass * dt;
    velo.x += force.x * iMdt * linearFactor.x;
    velo.y += force.y * iMdt * linearFactor.y;
    velo.z += force.z * iMdt * linearFactor.z;
    const e = invInertia.elements;
    const angularFactor = this.angularFactor;
    const tx = torque.x * angularFactor.x;
    const ty = torque.y * angularFactor.y;
    const tz = torque.z * angularFactor.z;
    angularVelo.x += dt * (e[0] * tx + e[1] * ty + e[2] * tz);
    angularVelo.y += dt * (e[3] * tx + e[4] * ty + e[5] * tz);
    angularVelo.z += dt * (e[6] * tx + e[7] * ty + e[8] * tz); // Use new velocity  - leap frog

    pos.x += velo.x * dt;
    pos.y += velo.y * dt;
    pos.z += velo.z * dt;
    quat.integrate(this.angularVelocity, dt, this.angularFactor, quat);

    if (quatNormalize) {
      if (quatNormalizeFast) {
        quat.normalizeFast();
      } else {
        quat.normalize();
      }
    }

    this.aabbNeedsUpdate = true; // Update world inertia

    this.updateInertiaWorld();
  }

}
Body.idCounter = 0;
Body.COLLIDE_EVENT_NAME = 'collide';
Body.DYNAMIC = BODY_TYPES.DYNAMIC;
Body.STATIC = BODY_TYPES.STATIC;
Body.KINEMATIC = BODY_TYPES.KINEMATIC;
Body.AWAKE = BODY_SLEEP_STATES.AWAKE;
Body.SLEEPY = BODY_SLEEP_STATES.SLEEPY;
Body.SLEEPING = BODY_SLEEP_STATES.SLEEPING;
Body.wakeupEvent = {
  type: 'wakeup'
};
Body.sleepyEvent = {
  type: 'sleepy'
};
Body.sleepEvent = {
  type: 'sleep'
};
const tmpVec = new Vec3();
const tmpQuat = new Quaternion();
const updateAABB_shapeAABB = new AABB();
const uiw_m1 = new Mat3();
const uiw_m2 = new Mat3();
const uiw_m3 = new Mat3();
const Body_applyForce_rotForce = new Vec3();
const Body_applyLocalForce_worldForce = new Vec3();
const Body_applyLocalForce_relativePointWorld = new Vec3();
const Body_applyImpulse_velo = new Vec3();
const Body_applyImpulse_rotVelo = new Vec3();
const Body_applyLocalImpulse_worldImpulse = new Vec3();
const Body_applyLocalImpulse_relativePoint = new Vec3();
const Body_updateMassProperties_halfExtents = new Vec3();

new Vec3();
new Vec3();
new Quaternion();
new Vec3();
new Vec3();
new Vec3();
new Vec3();

/**
 * Storage for Ray casting data
 */
class RaycastResult {
  /**
   * rayFromWorld
   */

  /**
   * rayToWorld
   */

  /**
   * hitNormalWorld
   */

  /**
   * hitPointWorld
   */

  /**
   * hasHit
   */

  /**
   * shape
   */

  /**
   * body
   */

  /**
   * The index of the hit triangle, if the hit shape was a trimesh
   */

  /**
   * Distance to the hit. Will be set to -1 if there was no hit
   */

  /**
   * If the ray should stop traversing the bodies
   */
  constructor() {
    this.rayFromWorld = new Vec3();
    this.rayToWorld = new Vec3();
    this.hitNormalWorld = new Vec3();
    this.hitPointWorld = new Vec3();
    this.hasHit = false;
    this.shape = null;
    this.body = null;
    this.hitFaceIndex = -1;
    this.distance = -1;
    this.shouldStop = false;
  }
  /**
   * Reset all result data.
   */


  reset() {
    this.rayFromWorld.setZero();
    this.rayToWorld.setZero();
    this.hitNormalWorld.setZero();
    this.hitPointWorld.setZero();
    this.hasHit = false;
    this.shape = null;
    this.body = null;
    this.hitFaceIndex = -1;
    this.distance = -1;
    this.shouldStop = false;
  }
  /**
   * abort
   */


  abort() {
    this.shouldStop = true;
  }
  /**
   * Set result data.
   */


  set(rayFromWorld, rayToWorld, hitNormalWorld, hitPointWorld, shape, body, distance) {
    this.rayFromWorld.copy(rayFromWorld);
    this.rayToWorld.copy(rayToWorld);
    this.hitNormalWorld.copy(hitNormalWorld);
    this.hitPointWorld.copy(hitPointWorld);
    this.shape = shape;
    this.body = body;
    this.distance = distance;
  }

}

let _Shape$types$SPHERE, _Shape$types$PLANE, _Shape$types$BOX, _Shape$types$CYLINDER, _Shape$types$CONVEXPO, _Shape$types$HEIGHTFI, _Shape$types$TRIMESH;

/**
 * RAY_MODES
 */
const RAY_MODES = {
  /** CLOSEST */
  CLOSEST: 1,

  /** ANY */
  ANY: 2,

  /** ALL */
  ALL: 4
};
/**
 * RayMode
 */

_Shape$types$SPHERE = Shape.types.SPHERE;
_Shape$types$PLANE = Shape.types.PLANE;
_Shape$types$BOX = Shape.types.BOX;
_Shape$types$CYLINDER = Shape.types.CYLINDER;
_Shape$types$CONVEXPO = Shape.types.CONVEXPOLYHEDRON;
_Shape$types$HEIGHTFI = Shape.types.HEIGHTFIELD;
_Shape$types$TRIMESH = Shape.types.TRIMESH;

/**
 * A line in 3D space that intersects bodies and return points.
 */
class Ray {
  /**
   * from
   */

  /**
   * to
   */

  /**
   * direction
   */

  /**
   * The precision of the ray. Used when checking parallelity etc.
   * @default 0.0001
   */

  /**
   * Set to `false` if you don't want the Ray to take `collisionResponse` flags into account on bodies and shapes.
   * @default true
   */

  /**
   * If set to `true`, the ray skips any hits with normal.dot(rayDirection) < 0.
   * @default false
   */

  /**
   * collisionFilterMask
   * @default -1
   */

  /**
   * collisionFilterGroup
   * @default -1
   */

  /**
   * The intersection mode. Should be Ray.ANY, Ray.ALL or Ray.CLOSEST.
   * @default RAY.ANY
   */

  /**
   * Current result object.
   */

  /**
   * Will be set to `true` during intersectWorld() if the ray hit anything.
   */

  /**
   * User-provided result callback. Will be used if mode is Ray.ALL.
   */

  /**
   * CLOSEST
   */

  /**
   * ANY
   */

  /**
   * ALL
   */
  get [_Shape$types$SPHERE]() {
    return this._intersectSphere;
  }

  get [_Shape$types$PLANE]() {
    return this._intersectPlane;
  }

  get [_Shape$types$BOX]() {
    return this._intersectBox;
  }

  get [_Shape$types$CYLINDER]() {
    return this._intersectConvex;
  }

  get [_Shape$types$CONVEXPO]() {
    return this._intersectConvex;
  }

  get [_Shape$types$HEIGHTFI]() {
    return this._intersectHeightfield;
  }

  get [_Shape$types$TRIMESH]() {
    return this._intersectTrimesh;
  }

  constructor(from, to) {
    if (from === void 0) {
      from = new Vec3();
    }

    if (to === void 0) {
      to = new Vec3();
    }

    this.from = from.clone();
    this.to = to.clone();
    this.direction = new Vec3();
    this.precision = 0.0001;
    this.checkCollisionResponse = true;
    this.skipBackfaces = false;
    this.collisionFilterMask = -1;
    this.collisionFilterGroup = -1;
    this.mode = Ray.ANY;
    this.result = new RaycastResult();
    this.hasHit = false;

    this.callback = result => {};
  }
  /**
   * Do itersection against all bodies in the given World.
   * @return True if the ray hit anything, otherwise false.
   */


  intersectWorld(world, options) {
    this.mode = options.mode || Ray.ANY;
    this.result = options.result || new RaycastResult();
    this.skipBackfaces = !!options.skipBackfaces;
    this.collisionFilterMask = typeof options.collisionFilterMask !== 'undefined' ? options.collisionFilterMask : -1;
    this.collisionFilterGroup = typeof options.collisionFilterGroup !== 'undefined' ? options.collisionFilterGroup : -1;
    this.checkCollisionResponse = typeof options.checkCollisionResponse !== 'undefined' ? options.checkCollisionResponse : true;

    if (options.from) {
      this.from.copy(options.from);
    }

    if (options.to) {
      this.to.copy(options.to);
    }

    this.callback = options.callback || (() => {});

    this.hasHit = false;
    this.result.reset();
    this.updateDirection();
    this.getAABB(tmpAABB$1);
    tmpArray.length = 0;
    world.broadphase.aabbQuery(world, tmpAABB$1, tmpArray);
    this.intersectBodies(tmpArray);
    return this.hasHit;
  }
  /**
   * Shoot a ray at a body, get back information about the hit.
   * @deprecated @param result set the result property of the Ray instead.
   */


  intersectBody(body, result) {
    if (result) {
      this.result = result;
      this.updateDirection();
    }

    const checkCollisionResponse = this.checkCollisionResponse;

    if (checkCollisionResponse && !body.collisionResponse) {
      return;
    }

    if ((this.collisionFilterGroup & body.collisionFilterMask) === 0 || (body.collisionFilterGroup & this.collisionFilterMask) === 0) {
      return;
    }

    const xi = intersectBody_xi;
    const qi = intersectBody_qi;

    for (let i = 0, N = body.shapes.length; i < N; i++) {
      const shape = body.shapes[i];

      if (checkCollisionResponse && !shape.collisionResponse) {
        continue; // Skip
      }

      body.quaternion.mult(body.shapeOrientations[i], qi);
      body.quaternion.vmult(body.shapeOffsets[i], xi);
      xi.vadd(body.position, xi);
      this.intersectShape(shape, qi, xi, body);

      if (this.result.shouldStop) {
        break;
      }
    }
  }
  /**
   * Shoot a ray at an array bodies, get back information about the hit.
   * @param bodies An array of Body objects.
   * @deprecated @param result set the result property of the Ray instead.
   *
   */


  intersectBodies(bodies, result) {
    if (result) {
      this.result = result;
      this.updateDirection();
    }

    for (let i = 0, l = bodies.length; !this.result.shouldStop && i < l; i++) {
      this.intersectBody(bodies[i]);
    }
  }
  /**
   * Updates the direction vector.
   */


  updateDirection() {
    this.to.vsub(this.from, this.direction);
    this.direction.normalize();
  }

  intersectShape(shape, quat, position, body) {
    const from = this.from; // Checking boundingSphere

    const distance = distanceFromIntersection(from, this.direction, position);

    if (distance > shape.boundingSphereRadius) {
      return;
    }

    const intersectMethod = this[shape.type];

    if (intersectMethod) {
      intersectMethod.call(this, shape, quat, position, body, shape);
    }
  }

  _intersectBox(box, quat, position, body, reportedShape) {
    return this._intersectConvex(box.convexPolyhedronRepresentation, quat, position, body, reportedShape);
  }

  _intersectPlane(shape, quat, position, body, reportedShape) {
    const from = this.from;
    const to = this.to;
    const direction = this.direction; // Get plane normal

    const worldNormal = new Vec3(0, 0, 1);
    quat.vmult(worldNormal, worldNormal);
    const len = new Vec3();
    from.vsub(position, len);
    const planeToFrom = len.dot(worldNormal);
    to.vsub(position, len);
    const planeToTo = len.dot(worldNormal);

    if (planeToFrom * planeToTo > 0) {
      // "from" and "to" are on the same side of the plane... bail out
      return;
    }

    if (from.distanceTo(to) < planeToFrom) {
      return;
    }

    const n_dot_dir = worldNormal.dot(direction);

    if (Math.abs(n_dot_dir) < this.precision) {
      // No intersection
      return;
    }

    const planePointToFrom = new Vec3();
    const dir_scaled_with_t = new Vec3();
    const hitPointWorld = new Vec3();
    from.vsub(position, planePointToFrom);
    const t = -worldNormal.dot(planePointToFrom) / n_dot_dir;
    direction.scale(t, dir_scaled_with_t);
    from.vadd(dir_scaled_with_t, hitPointWorld);
    this.reportIntersection(worldNormal, hitPointWorld, reportedShape, body, -1);
  }
  /**
   * Get the world AABB of the ray.
   */


  getAABB(aabb) {
    const {
      lowerBound,
      upperBound
    } = aabb;
    const to = this.to;
    const from = this.from;
    lowerBound.x = Math.min(to.x, from.x);
    lowerBound.y = Math.min(to.y, from.y);
    lowerBound.z = Math.min(to.z, from.z);
    upperBound.x = Math.max(to.x, from.x);
    upperBound.y = Math.max(to.y, from.y);
    upperBound.z = Math.max(to.z, from.z);
  }

  _intersectHeightfield(shape, quat, position, body, reportedShape) {
    shape.data;
    shape.elementSize; // Convert the ray to local heightfield coordinates

    const localRay = intersectHeightfield_localRay; //new Ray(this.from, this.to);

    localRay.from.copy(this.from);
    localRay.to.copy(this.to);
    Transform.pointToLocalFrame(position, quat, localRay.from, localRay.from);
    Transform.pointToLocalFrame(position, quat, localRay.to, localRay.to);
    localRay.updateDirection(); // Get the index of the data points to test against

    const index = intersectHeightfield_index;
    let iMinX;
    let iMinY;
    let iMaxX;
    let iMaxY; // Set to max

    iMinX = iMinY = 0;
    iMaxX = iMaxY = shape.data.length - 1;
    const aabb = new AABB();
    localRay.getAABB(aabb);
    shape.getIndexOfPosition(aabb.lowerBound.x, aabb.lowerBound.y, index, true);
    iMinX = Math.max(iMinX, index[0]);
    iMinY = Math.max(iMinY, index[1]);
    shape.getIndexOfPosition(aabb.upperBound.x, aabb.upperBound.y, index, true);
    iMaxX = Math.min(iMaxX, index[0] + 1);
    iMaxY = Math.min(iMaxY, index[1] + 1);

    for (let i = iMinX; i < iMaxX; i++) {
      for (let j = iMinY; j < iMaxY; j++) {
        if (this.result.shouldStop) {
          return;
        }

        shape.getAabbAtIndex(i, j, aabb);

        if (!aabb.overlapsRay(localRay)) {
          continue;
        } // Lower triangle


        shape.getConvexTrianglePillar(i, j, false);
        Transform.pointToWorldFrame(position, quat, shape.pillarOffset, worldPillarOffset);

        this._intersectConvex(shape.pillarConvex, quat, worldPillarOffset, body, reportedShape, intersectConvexOptions);

        if (this.result.shouldStop) {
          return;
        } // Upper triangle


        shape.getConvexTrianglePillar(i, j, true);
        Transform.pointToWorldFrame(position, quat, shape.pillarOffset, worldPillarOffset);

        this._intersectConvex(shape.pillarConvex, quat, worldPillarOffset, body, reportedShape, intersectConvexOptions);
      }
    }
  }

  _intersectSphere(sphere, quat, position, body, reportedShape) {
    const from = this.from;
    const to = this.to;
    const r = sphere.radius;
    const a = (to.x - from.x) ** 2 + (to.y - from.y) ** 2 + (to.z - from.z) ** 2;
    const b = 2 * ((to.x - from.x) * (from.x - position.x) + (to.y - from.y) * (from.y - position.y) + (to.z - from.z) * (from.z - position.z));
    const c = (from.x - position.x) ** 2 + (from.y - position.y) ** 2 + (from.z - position.z) ** 2 - r ** 2;
    const delta = b ** 2 - 4 * a * c;
    const intersectionPoint = Ray_intersectSphere_intersectionPoint;
    const normal = Ray_intersectSphere_normal;

    if (delta < 0) {
      // No intersection
      return;
    } else if (delta === 0) {
      // single intersection point
      from.lerp(to, delta, intersectionPoint);
      intersectionPoint.vsub(position, normal);
      normal.normalize();
      this.reportIntersection(normal, intersectionPoint, reportedShape, body, -1);
    } else {
      const d1 = (-b - Math.sqrt(delta)) / (2 * a);
      const d2 = (-b + Math.sqrt(delta)) / (2 * a);

      if (d1 >= 0 && d1 <= 1) {
        from.lerp(to, d1, intersectionPoint);
        intersectionPoint.vsub(position, normal);
        normal.normalize();
        this.reportIntersection(normal, intersectionPoint, reportedShape, body, -1);
      }

      if (this.result.shouldStop) {
        return;
      }

      if (d2 >= 0 && d2 <= 1) {
        from.lerp(to, d2, intersectionPoint);
        intersectionPoint.vsub(position, normal);
        normal.normalize();
        this.reportIntersection(normal, intersectionPoint, reportedShape, body, -1);
      }
    }
  }

  _intersectConvex(shape, quat, position, body, reportedShape, options) {
    const normal = intersectConvex_normal;
    const vector = intersectConvex_vector;
    const faceList = options && options.faceList || null; // Checking faces

    const faces = shape.faces;
    const vertices = shape.vertices;
    const normals = shape.faceNormals;
    const direction = this.direction;
    const from = this.from;
    const to = this.to;
    const fromToDistance = from.distanceTo(to);
    const Nfaces = faceList ? faceList.length : faces.length;
    const result = this.result;

    for (let j = 0; !result.shouldStop && j < Nfaces; j++) {
      const fi = faceList ? faceList[j] : j;
      const face = faces[fi];
      const faceNormal = normals[fi];
      const q = quat;
      const x = position; // determine if ray intersects the plane of the face
      // note: this works regardless of the direction of the face normal
      // Get plane point in world coordinates...

      vector.copy(vertices[face[0]]);
      q.vmult(vector, vector);
      vector.vadd(x, vector); // ...but make it relative to the ray from. We'll fix this later.

      vector.vsub(from, vector); // Get plane normal

      q.vmult(faceNormal, normal); // If this dot product is negative, we have something interesting

      const dot = direction.dot(normal); // Bail out if ray and plane are parallel

      if (Math.abs(dot) < this.precision) {
        continue;
      } // calc distance to plane


      const scalar = normal.dot(vector) / dot; // if negative distance, then plane is behind ray

      if (scalar < 0) {
        continue;
      } // if (dot < 0) {
      // Intersection point is from + direction * scalar


      direction.scale(scalar, intersectPoint);
      intersectPoint.vadd(from, intersectPoint); // a is the point we compare points b and c with.

      a.copy(vertices[face[0]]);
      q.vmult(a, a);
      x.vadd(a, a);

      for (let i = 1; !result.shouldStop && i < face.length - 1; i++) {
        // Transform 3 vertices to world coords
        b.copy(vertices[face[i]]);
        c.copy(vertices[face[i + 1]]);
        q.vmult(b, b);
        q.vmult(c, c);
        x.vadd(b, b);
        x.vadd(c, c);
        const distance = intersectPoint.distanceTo(from);

        if (!(Ray.pointInTriangle(intersectPoint, a, b, c) || Ray.pointInTriangle(intersectPoint, b, a, c)) || distance > fromToDistance) {
          continue;
        }

        this.reportIntersection(normal, intersectPoint, reportedShape, body, fi);
      } // }

    }
  }
  /**
   * @todo Optimize by transforming the world to local space first.
   * @todo Use Octree lookup
   */


  _intersectTrimesh(mesh, quat, position, body, reportedShape, options) {
    const normal = intersectTrimesh_normal;
    const triangles = intersectTrimesh_triangles;
    const treeTransform = intersectTrimesh_treeTransform;
    const vector = intersectConvex_vector;
    const localDirection = intersectTrimesh_localDirection;
    const localFrom = intersectTrimesh_localFrom;
    const localTo = intersectTrimesh_localTo;
    const worldIntersectPoint = intersectTrimesh_worldIntersectPoint;
    const worldNormal = intersectTrimesh_worldNormal; // Checking faces

    const indices = mesh.indices;
    mesh.vertices; // const normals = mesh.faceNormals

    const from = this.from;
    const to = this.to;
    const direction = this.direction;
    treeTransform.position.copy(position);
    treeTransform.quaternion.copy(quat); // Transform ray to local space!

    Transform.vectorToLocalFrame(position, quat, direction, localDirection);
    Transform.pointToLocalFrame(position, quat, from, localFrom);
    Transform.pointToLocalFrame(position, quat, to, localTo);
    localTo.x *= mesh.scale.x;
    localTo.y *= mesh.scale.y;
    localTo.z *= mesh.scale.z;
    localFrom.x *= mesh.scale.x;
    localFrom.y *= mesh.scale.y;
    localFrom.z *= mesh.scale.z;
    localTo.vsub(localFrom, localDirection);
    localDirection.normalize();
    const fromToDistanceSquared = localFrom.distanceSquared(localTo);
    mesh.tree.rayQuery(this, treeTransform, triangles);

    for (let i = 0, N = triangles.length; !this.result.shouldStop && i !== N; i++) {
      const trianglesIndex = triangles[i];
      mesh.getNormal(trianglesIndex, normal); // determine if ray intersects the plane of the face
      // note: this works regardless of the direction of the face normal
      // Get plane point in world coordinates...

      mesh.getVertex(indices[trianglesIndex * 3], a); // ...but make it relative to the ray from. We'll fix this later.

      a.vsub(localFrom, vector); // If this dot product is negative, we have something interesting

      const dot = localDirection.dot(normal); // Bail out if ray and plane are parallel
      // if (Math.abs( dot ) < this.precision){
      //     continue;
      // }
      // calc distance to plane

      const scalar = normal.dot(vector) / dot; // if negative distance, then plane is behind ray

      if (scalar < 0) {
        continue;
      } // Intersection point is from + direction * scalar


      localDirection.scale(scalar, intersectPoint);
      intersectPoint.vadd(localFrom, intersectPoint); // Get triangle vertices

      mesh.getVertex(indices[trianglesIndex * 3 + 1], b);
      mesh.getVertex(indices[trianglesIndex * 3 + 2], c);
      const squaredDistance = intersectPoint.distanceSquared(localFrom);

      if (!(Ray.pointInTriangle(intersectPoint, b, a, c) || Ray.pointInTriangle(intersectPoint, a, b, c)) || squaredDistance > fromToDistanceSquared) {
        continue;
      } // transform intersectpoint and normal to world


      Transform.vectorToWorldFrame(quat, normal, worldNormal);
      Transform.pointToWorldFrame(position, quat, intersectPoint, worldIntersectPoint);
      this.reportIntersection(worldNormal, worldIntersectPoint, reportedShape, body, trianglesIndex);
    }

    triangles.length = 0;
  }
  /**
   * @return True if the intersections should continue
   */


  reportIntersection(normal, hitPointWorld, shape, body, hitFaceIndex) {
    const from = this.from;
    const to = this.to;
    const distance = from.distanceTo(hitPointWorld);
    const result = this.result; // Skip back faces?

    if (this.skipBackfaces && normal.dot(this.direction) > 0) {
      return;
    }

    result.hitFaceIndex = typeof hitFaceIndex !== 'undefined' ? hitFaceIndex : -1;

    switch (this.mode) {
      case Ray.ALL:
        this.hasHit = true;
        result.set(from, to, normal, hitPointWorld, shape, body, distance);
        result.hasHit = true;
        this.callback(result);
        break;

      case Ray.CLOSEST:
        // Store if closer than current closest
        if (distance < result.distance || !result.hasHit) {
          this.hasHit = true;
          result.hasHit = true;
          result.set(from, to, normal, hitPointWorld, shape, body, distance);
        }

        break;

      case Ray.ANY:
        // Report and stop.
        this.hasHit = true;
        result.hasHit = true;
        result.set(from, to, normal, hitPointWorld, shape, body, distance);
        result.shouldStop = true;
        break;
    }
  }
  /**
   * As per "Barycentric Technique" as named
   * {@link https://www.blackpawn.com/texts/pointinpoly/default.html here} but without the division
   */


  static pointInTriangle(p, a, b, c) {
    c.vsub(a, v0);
    b.vsub(a, v1);
    p.vsub(a, v2);
    const dot00 = v0.dot(v0);
    const dot01 = v0.dot(v1);
    const dot02 = v0.dot(v2);
    const dot11 = v1.dot(v1);
    const dot12 = v1.dot(v2);
    let u;
    let v;
    return (u = dot11 * dot02 - dot01 * dot12) >= 0 && (v = dot00 * dot12 - dot01 * dot02) >= 0 && u + v < dot00 * dot11 - dot01 * dot01;
  }

}
Ray.CLOSEST = RAY_MODES.CLOSEST;
Ray.ANY = RAY_MODES.ANY;
Ray.ALL = RAY_MODES.ALL;
const tmpAABB$1 = new AABB();
const tmpArray = [];
const v1 = new Vec3();
const v2 = new Vec3();
const intersectBody_xi = new Vec3();
const intersectBody_qi = new Quaternion();
const intersectPoint = new Vec3();
const a = new Vec3();
const b = new Vec3();
const c = new Vec3();
new Vec3();
new RaycastResult();
const intersectConvexOptions = {
  faceList: [0]
};
const worldPillarOffset = new Vec3();
const intersectHeightfield_localRay = new Ray();
const intersectHeightfield_index = [];
const Ray_intersectSphere_intersectionPoint = new Vec3();
const Ray_intersectSphere_normal = new Vec3();
const intersectConvex_normal = new Vec3();
new Vec3();
new Vec3();
const intersectConvex_vector = new Vec3();
const intersectTrimesh_normal = new Vec3();
const intersectTrimesh_localDirection = new Vec3();
const intersectTrimesh_localFrom = new Vec3();
const intersectTrimesh_localTo = new Vec3();
const intersectTrimesh_worldNormal = new Vec3();
const intersectTrimesh_worldIntersectPoint = new Vec3();
new AABB();
const intersectTrimesh_triangles = [];
const intersectTrimesh_treeTransform = new Transform();
const v0 = new Vec3();
const intersect = new Vec3();

function distanceFromIntersection(from, direction, position) {
  // v0 is vector from from to position
  position.vsub(from, v0);
  const dot = v0.dot(direction); // intersect = direction*dot + from

  direction.scale(dot, intersect);
  intersect.vadd(from, intersect);
  const distance = position.distanceTo(intersect);
  return distance;
}

class Utils {
  /**
   * Extend an options object with default values.
   * @param options The options object. May be falsy: in this case, a new object is created and returned.
   * @param defaults An object containing default values.
   * @return The modified options object.
   */
  static defaults(options, defaults) {
    if (options === void 0) {
      options = {};
    }

    for (let key in defaults) {
      if (!(key in options)) {
        options[key] = defaults[key];
      }
    }

    return options;
  }

}

/**
 * Constraint base class
 */
class Constraint {
  /**
   * Equations to be solved in this constraint.
   */

  /**
   * Body A.
   */

  /**
   * Body B.
   */

  /**
   * Set to false if you don't want the bodies to collide when they are connected.
   */
  constructor(bodyA, bodyB, options) {
    if (options === void 0) {
      options = {};
    }

    options = Utils.defaults(options, {
      collideConnected: true,
      wakeUpBodies: true
    });
    this.equations = [];
    this.bodyA = bodyA;
    this.bodyB = bodyB;
    this.id = Constraint.idCounter++;
    this.collideConnected = options.collideConnected;

    if (options.wakeUpBodies) {
      if (bodyA) {
        bodyA.wakeUp();
      }

      if (bodyB) {
        bodyB.wakeUp();
      }
    }
  }
  /**
   * Update all the equations with data.
   */


  update() {
    throw new Error('method update() not implmemented in this Constraint subclass!');
  }
  /**
   * Enables all equations in the constraint.
   */


  enable() {
    const eqs = this.equations;

    for (let i = 0; i < eqs.length; i++) {
      eqs[i].enabled = true;
    }
  }
  /**
   * Disables all equations in the constraint.
   */


  disable() {
    const eqs = this.equations;

    for (let i = 0; i < eqs.length; i++) {
      eqs[i].enabled = false;
    }
  }

}
Constraint.idCounter = 0;

/**
 * An element containing 6 entries, 3 spatial and 3 rotational degrees of freedom.
 */

class JacobianElement {
  /**
   * spatial
   */

  /**
   * rotational
   */
  constructor() {
    this.spatial = new Vec3();
    this.rotational = new Vec3();
  }
  /**
   * Multiply with other JacobianElement
   */


  multiplyElement(element) {
    return element.spatial.dot(this.spatial) + element.rotational.dot(this.rotational);
  }
  /**
   * Multiply with two vectors
   */


  multiplyVectors(spatial, rotational) {
    return spatial.dot(this.spatial) + rotational.dot(this.rotational);
  }

}

/**
 * Equation base class.
 *
 * `a`, `b` and `eps` are {@link https://www8.cs.umu.se/kurser/5DV058/VT15/lectures/SPOOKlabnotes.pdf SPOOK} parameters that default to `0.0`. See {@link https://github.com/schteppe/cannon.js/issues/238#issuecomment-147172327 this exchange} for more details on Cannon's physics implementation.
 */
class Equation {
  /**
   * Minimum (read: negative max) force to be applied by the constraint.
   */

  /**
   * Maximum (read: positive max) force to be applied by the constraint.
   */

  /**
   * SPOOK parameter
   */

  /**
   * SPOOK parameter
   */

  /**
   * SPOOK parameter
   */

  /**
   * A number, proportional to the force added to the bodies.
   */
  constructor(bi, bj, minForce, maxForce) {
    if (minForce === void 0) {
      minForce = -1e6;
    }

    if (maxForce === void 0) {
      maxForce = 1e6;
    }

    this.id = Equation.idCounter++;
    this.minForce = minForce;
    this.maxForce = maxForce;
    this.bi = bi;
    this.bj = bj;
    this.a = 0.0; // SPOOK parameter

    this.b = 0.0; // SPOOK parameter

    this.eps = 0.0; // SPOOK parameter

    this.jacobianElementA = new JacobianElement();
    this.jacobianElementB = new JacobianElement();
    this.enabled = true;
    this.multiplier = 0;
    this.setSpookParams(1e7, 4, 1 / 60); // Set typical spook params
  }
  /**
   * Recalculates a, b, and eps.
   *
   * The Equation constructor sets typical SPOOK parameters as such:
   * * `stiffness` = 1e7
   * * `relaxation` = 4
   * * `timeStep`= 1 / 60, _note the hardcoded refresh rate._
   */


  setSpookParams(stiffness, relaxation, timeStep) {
    const d = relaxation;
    const k = stiffness;
    const h = timeStep;
    this.a = 4.0 / (h * (1 + 4 * d));
    this.b = 4.0 * d / (1 + 4 * d);
    this.eps = 4.0 / (h * h * k * (1 + 4 * d));
  }
  /**
   * Computes the right hand side of the SPOOK equation
   */


  computeB(a, b, h) {
    const GW = this.computeGW();
    const Gq = this.computeGq();
    const GiMf = this.computeGiMf();
    return -Gq * a - GW * b - GiMf * h;
  }
  /**
   * Computes G*q, where q are the generalized body coordinates
   */


  computeGq() {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const xi = bi.position;
    const xj = bj.position;
    return GA.spatial.dot(xi) + GB.spatial.dot(xj);
  }
  /**
   * Computes G*W, where W are the body velocities
   */


  computeGW() {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const vi = bi.velocity;
    const vj = bj.velocity;
    const wi = bi.angularVelocity;
    const wj = bj.angularVelocity;
    return GA.multiplyVectors(vi, wi) + GB.multiplyVectors(vj, wj);
  }
  /**
   * Computes G*Wlambda, where W are the body velocities
   */


  computeGWlambda() {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const vi = bi.vlambda;
    const vj = bj.vlambda;
    const wi = bi.wlambda;
    const wj = bj.wlambda;
    return GA.multiplyVectors(vi, wi) + GB.multiplyVectors(vj, wj);
  }
  /**
   * Computes G*inv(M)*f, where M is the mass matrix with diagonal blocks for each body, and f are the forces on the bodies.
   */


  computeGiMf() {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const fi = bi.force;
    const ti = bi.torque;
    const fj = bj.force;
    const tj = bj.torque;
    const invMassi = bi.invMassSolve;
    const invMassj = bj.invMassSolve;
    fi.scale(invMassi, iMfi);
    fj.scale(invMassj, iMfj);
    bi.invInertiaWorldSolve.vmult(ti, invIi_vmult_taui);
    bj.invInertiaWorldSolve.vmult(tj, invIj_vmult_tauj);
    return GA.multiplyVectors(iMfi, invIi_vmult_taui) + GB.multiplyVectors(iMfj, invIj_vmult_tauj);
  }
  /**
   * Computes G*inv(M)*G'
   */


  computeGiMGt() {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const invMassi = bi.invMassSolve;
    const invMassj = bj.invMassSolve;
    const invIi = bi.invInertiaWorldSolve;
    const invIj = bj.invInertiaWorldSolve;
    let result = invMassi + invMassj;
    invIi.vmult(GA.rotational, tmp);
    result += tmp.dot(GA.rotational);
    invIj.vmult(GB.rotational, tmp);
    result += tmp.dot(GB.rotational);
    return result;
  }
  /**
   * Add constraint velocity to the bodies.
   */


  addToWlambda(deltalambda) {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const temp = addToWlambda_temp; // Add to linear velocity
    // v_lambda += inv(M) * delta_lamba * G

    bi.vlambda.addScaledVector(bi.invMassSolve * deltalambda, GA.spatial, bi.vlambda);
    bj.vlambda.addScaledVector(bj.invMassSolve * deltalambda, GB.spatial, bj.vlambda); // Add to angular velocity

    bi.invInertiaWorldSolve.vmult(GA.rotational, temp);
    bi.wlambda.addScaledVector(deltalambda, temp, bi.wlambda);
    bj.invInertiaWorldSolve.vmult(GB.rotational, temp);
    bj.wlambda.addScaledVector(deltalambda, temp, bj.wlambda);
  }
  /**
   * Compute the denominator part of the SPOOK equation: C = G*inv(M)*G' + eps
   */


  computeC() {
    return this.computeGiMGt() + this.eps;
  }

}
Equation.idCounter = 0;
const iMfi = new Vec3();
const iMfj = new Vec3();
const invIi_vmult_taui = new Vec3();
const invIj_vmult_tauj = new Vec3();
const tmp = new Vec3();
const addToWlambda_temp = new Vec3();

/**
 * Contact/non-penetration constraint equation
 */
class ContactEquation extends Equation {
  /**
   * "bounciness": u1 = -e*u0
   */

  /**
   * World-oriented vector that goes from the center of bi to the contact point.
   */

  /**
   * World-oriented vector that starts in body j position and goes to the contact point.
   */

  /**
   * Contact normal, pointing out of body i.
   */
  constructor(bodyA, bodyB, maxForce) {
    if (maxForce === void 0) {
      maxForce = 1e6;
    }

    super(bodyA, bodyB, 0, maxForce);
    this.restitution = 0.0;
    this.ri = new Vec3();
    this.rj = new Vec3();
    this.ni = new Vec3();
  }

  computeB(h) {
    const a = this.a;
    const b = this.b;
    const bi = this.bi;
    const bj = this.bj;
    const ri = this.ri;
    const rj = this.rj;
    const rixn = ContactEquation_computeB_temp1;
    const rjxn = ContactEquation_computeB_temp2;
    const vi = bi.velocity;
    const wi = bi.angularVelocity;
    bi.force;
    bi.torque;
    const vj = bj.velocity;
    const wj = bj.angularVelocity;
    bj.force;
    bj.torque;
    const penetrationVec = ContactEquation_computeB_temp3;
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const n = this.ni; // Caluclate cross products

    ri.cross(n, rixn);
    rj.cross(n, rjxn); // g = xj+rj -(xi+ri)
    // G = [ -ni  -rixn  ni  rjxn ]

    n.negate(GA.spatial);
    rixn.negate(GA.rotational);
    GB.spatial.copy(n);
    GB.rotational.copy(rjxn); // Calculate the penetration vector

    penetrationVec.copy(bj.position);
    penetrationVec.vadd(rj, penetrationVec);
    penetrationVec.vsub(bi.position, penetrationVec);
    penetrationVec.vsub(ri, penetrationVec);
    const g = n.dot(penetrationVec); // Compute iteration

    const ePlusOne = this.restitution + 1;
    const GW = ePlusOne * vj.dot(n) - ePlusOne * vi.dot(n) + wj.dot(rjxn) - wi.dot(rixn);
    const GiMf = this.computeGiMf();
    const B = -g * a - GW * b - h * GiMf;
    return B;
  }
  /**
   * Get the current relative velocity in the contact point.
   */


  getImpactVelocityAlongNormal() {
    const vi = ContactEquation_getImpactVelocityAlongNormal_vi;
    const vj = ContactEquation_getImpactVelocityAlongNormal_vj;
    const xi = ContactEquation_getImpactVelocityAlongNormal_xi;
    const xj = ContactEquation_getImpactVelocityAlongNormal_xj;
    const relVel = ContactEquation_getImpactVelocityAlongNormal_relVel;
    this.bi.position.vadd(this.ri, xi);
    this.bj.position.vadd(this.rj, xj);
    this.bi.getVelocityAtWorldPoint(xi, vi);
    this.bj.getVelocityAtWorldPoint(xj, vj);
    vi.vsub(vj, relVel);
    return this.ni.dot(relVel);
  }

}
const ContactEquation_computeB_temp1 = new Vec3(); // Temp vectors

const ContactEquation_computeB_temp2 = new Vec3();
const ContactEquation_computeB_temp3 = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_vi = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_vj = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_xi = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_xj = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_relVel = new Vec3();

/**
 * Connects two bodies at given offset points.
 * @example
 *     const bodyA = new Body({ mass: 1 })
 *     const bodyB = new Body({ mass: 1 })
 *     bodyA.position.set(-1, 0, 0)
 *     bodyB.position.set(1, 0, 0)
 *     bodyA.addShape(shapeA)
 *     bodyB.addShape(shapeB)
 *     world.addBody(bodyA)
 *     world.addBody(bodyB)
 *     const localPivotA = new Vec3(1, 0, 0)
 *     const localPivotB = new Vec3(-1, 0, 0)
 *     const constraint = new PointToPointConstraint(bodyA, localPivotA, bodyB, localPivotB)
 *     world.addConstraint(constraint)
 */
class PointToPointConstraint extends Constraint {
  /**
   * Pivot, defined locally in bodyA.
   */

  /**
   * Pivot, defined locally in bodyB.
   */

  /**
   * @param pivotA The point relative to the center of mass of bodyA which bodyA is constrained to.
   * @param bodyB Body that will be constrained in a similar way to the same point as bodyA. We will therefore get a link between bodyA and bodyB. If not specified, bodyA will be constrained to a static point.
   * @param pivotB The point relative to the center of mass of bodyB which bodyB is constrained to.
   * @param maxForce The maximum force that should be applied to constrain the bodies.
   */
  constructor(bodyA, pivotA, bodyB, pivotB, maxForce) {
    if (pivotA === void 0) {
      pivotA = new Vec3();
    }

    if (pivotB === void 0) {
      pivotB = new Vec3();
    }

    if (maxForce === void 0) {
      maxForce = 1e6;
    }

    super(bodyA, bodyB);
    this.pivotA = pivotA.clone();
    this.pivotB = pivotB.clone();
    const x = this.equationX = new ContactEquation(bodyA, bodyB);
    const y = this.equationY = new ContactEquation(bodyA, bodyB);
    const z = this.equationZ = new ContactEquation(bodyA, bodyB); // Equations to be fed to the solver

    this.equations.push(x, y, z); // Make the equations bidirectional

    x.minForce = y.minForce = z.minForce = -maxForce;
    x.maxForce = y.maxForce = z.maxForce = maxForce;
    x.ni.set(1, 0, 0);
    y.ni.set(0, 1, 0);
    z.ni.set(0, 0, 1);
  }

  update() {
    const bodyA = this.bodyA;
    const bodyB = this.bodyB;
    const x = this.equationX;
    const y = this.equationY;
    const z = this.equationZ; // Rotate the pivots to world space

    bodyA.quaternion.vmult(this.pivotA, x.ri);
    bodyB.quaternion.vmult(this.pivotB, x.rj);
    y.ri.copy(x.ri);
    y.rj.copy(x.rj);
    z.ri.copy(x.ri);
    z.rj.copy(x.rj);
  }

}
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Ray();
new Vec3();
new Vec3();
new Vec3();
[new Vec3(1, 0, 0), new Vec3(0, 1, 0), new Vec3(0, 0, 1)];
new Vec3();
new Vec3();
new Vec3();
new Vec3();

new Vec3();
new Vec3();
new Vec3();
new Vec3();

new Vec3();
new Vec3();
new Vec3(); // bilateral constraint between two dynamic objects

/**
 * Spherical shape
 * @example
 *     const radius = 1
 *     const sphereShape = new CANNON.Sphere(radius)
 *     const sphereBody = new CANNON.Body({ mass: 1, shape: sphereShape })
 *     world.addBody(sphereBody)
 */
class Sphere extends Shape {
  /**
   * The radius of the sphere.
   */

  /**
   *
   * @param radius The radius of the sphere, a non-negative number.
   */
  constructor(radius) {
    super({
      type: Shape.types.SPHERE
    });
    this.radius = radius !== undefined ? radius : 1.0;

    if (this.radius < 0) {
      throw new Error('The sphere radius cannot be negative.');
    }

    this.updateBoundingSphereRadius();
  }
  /** calculateLocalInertia */


  calculateLocalInertia(mass, target) {
    if (target === void 0) {
      target = new Vec3();
    }

    const I = 2.0 * mass * this.radius * this.radius / 5.0;
    target.x = I;
    target.y = I;
    target.z = I;
    return target;
  }
  /** volume */


  volume() {
    return 4.0 * Math.PI * Math.pow(this.radius, 3) / 3.0;
  }

  updateBoundingSphereRadius() {
    this.boundingSphereRadius = this.radius;
  }

  calculateWorldAABB(pos, quat, min, max) {
    const r = this.radius;
    const axes = ['x', 'y', 'z'];

    for (let i = 0; i < axes.length; i++) {
      const ax = axes[i];
      min[ax] = pos[ax] - r;
      max[ax] = pos[ax] + r;
    }
  }

}
new Vec3();
new Vec3();
new Vec3(); // Temp vectors for calculation

new Vec3(); // Relative velocity

new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();

/**
 * Cylinder class.
 * @example
 *     const radiusTop = 0.5
 *     const radiusBottom = 0.5
 *     const height = 2
 *     const numSegments = 12
 *     const cylinderShape = new CANNON.Cylinder(radiusTop, radiusBottom, height, numSegments)
 *     const cylinderBody = new CANNON.Body({ mass: 1, shape: cylinderShape })
 *     world.addBody(cylinderBody)
 */

class Cylinder extends ConvexPolyhedron {
  /** The radius of the top of the Cylinder. */

  /** The radius of the bottom of the Cylinder. */

  /** The height of the Cylinder. */

  /** The number of segments to build the cylinder out of. */

  /**
   * @param radiusTop The radius of the top of the Cylinder.
   * @param radiusBottom The radius of the bottom of the Cylinder.
   * @param height The height of the Cylinder.
   * @param numSegments The number of segments to build the cylinder out of.
   */
  constructor(radiusTop, radiusBottom, height, numSegments) {
    if (radiusTop === void 0) {
      radiusTop = 1;
    }

    if (radiusBottom === void 0) {
      radiusBottom = 1;
    }

    if (height === void 0) {
      height = 1;
    }

    if (numSegments === void 0) {
      numSegments = 8;
    }

    if (radiusTop < 0) {
      throw new Error('The cylinder radiusTop cannot be negative.');
    }

    if (radiusBottom < 0) {
      throw new Error('The cylinder radiusBottom cannot be negative.');
    }

    const N = numSegments;
    const vertices = [];
    const axes = [];
    const faces = [];
    const bottomface = [];
    const topface = [];
    const cos = Math.cos;
    const sin = Math.sin; // First bottom point

    vertices.push(new Vec3(-radiusBottom * sin(0), -height * 0.5, radiusBottom * cos(0)));
    bottomface.push(0); // First top point

    vertices.push(new Vec3(-radiusTop * sin(0), height * 0.5, radiusTop * cos(0)));
    topface.push(1);

    for (let i = 0; i < N; i++) {
      const theta = 2 * Math.PI / N * (i + 1);
      const thetaN = 2 * Math.PI / N * (i + 0.5);

      if (i < N - 1) {
        // Bottom
        vertices.push(new Vec3(-radiusBottom * sin(theta), -height * 0.5, radiusBottom * cos(theta)));
        bottomface.push(2 * i + 2); // Top

        vertices.push(new Vec3(-radiusTop * sin(theta), height * 0.5, radiusTop * cos(theta)));
        topface.push(2 * i + 3); // Face

        faces.push([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2]);
      } else {
        faces.push([2 * i, 2 * i + 1, 1, 0]); // Connect
      } // Axis: we can cut off half of them if we have even number of segments


      if (N % 2 === 1 || i < N / 2) {
        axes.push(new Vec3(-sin(thetaN), 0, cos(thetaN)));
      }
    }

    faces.push(bottomface);
    axes.push(new Vec3(0, 1, 0)); // Reorder top face

    const temp = [];

    for (let i = 0; i < topface.length; i++) {
      temp.push(topface[topface.length - i - 1]);
    }

    faces.push(temp);
    super({
      vertices,
      faces,
      axes
    });
    this.type = Shape.types.CYLINDER;
    this.radiusTop = radiusTop;
    this.radiusBottom = radiusBottom;
    this.height = height;
    this.numSegments = numSegments;
  }

}
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3(); // from https://en.wikipedia.org/wiki/Barycentric_coordinate_system
new Vec3();
new AABB();
new Vec3();
new AABB();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new AABB();
new Vec3();
new Transform();
new AABB();

Body.STATIC;

// Naming rule: based of the order in SHAPE_TYPES,
// the first part of the method is formed by the
// shape type that comes before, in the second part
// there is the shape type that comes after in the SHAPE_TYPES list
({
  sphereSphere: Shape.types.SPHERE,
  spherePlane: Shape.types.SPHERE | Shape.types.PLANE,
  boxBox: Shape.types.BOX | Shape.types.BOX,
  sphereBox: Shape.types.SPHERE | Shape.types.BOX,
  planeBox: Shape.types.PLANE | Shape.types.BOX,
  convexConvex: Shape.types.CONVEXPOLYHEDRON,
  sphereConvex: Shape.types.SPHERE | Shape.types.CONVEXPOLYHEDRON,
  planeConvex: Shape.types.PLANE | Shape.types.CONVEXPOLYHEDRON,
  boxConvex: Shape.types.BOX | Shape.types.CONVEXPOLYHEDRON,
  sphereHeightfield: Shape.types.SPHERE | Shape.types.HEIGHTFIELD,
  boxHeightfield: Shape.types.BOX | Shape.types.HEIGHTFIELD,
  convexHeightfield: Shape.types.CONVEXPOLYHEDRON | Shape.types.HEIGHTFIELD,
  sphereParticle: Shape.types.PARTICLE | Shape.types.SPHERE,
  planeParticle: Shape.types.PLANE | Shape.types.PARTICLE,
  boxParticle: Shape.types.BOX | Shape.types.PARTICLE,
  convexParticle: Shape.types.PARTICLE | Shape.types.CONVEXPOLYHEDRON,
  cylinderCylinder: Shape.types.CYLINDER,
  sphereCylinder: Shape.types.SPHERE | Shape.types.CYLINDER,
  planeCylinder: Shape.types.PLANE | Shape.types.CYLINDER,
  boxCylinder: Shape.types.BOX | Shape.types.CYLINDER,
  convexCylinder: Shape.types.CONVEXPOLYHEDRON | Shape.types.CYLINDER,
  heightfieldCylinder: Shape.types.HEIGHTFIELD | Shape.types.CYLINDER,
  particleCylinder: Shape.types.PARTICLE | Shape.types.CYLINDER,
  sphereTrimesh: Shape.types.SPHERE | Shape.types.TRIMESH,
  planeTrimesh: Shape.types.PLANE | Shape.types.TRIMESH
});
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Quaternion();
new Quaternion();

new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new AABB();
new Vec3();
new Vec3(); // See http://bulletphysics.com/Bullet/BulletFull/SphereTriangleDetector_8cpp_source.html

new Vec3();
new Vec3();
new Vec3();

new Vec3();
new Vec3();
new Vec3();
new Vec3();
[new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3()];
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3(); // WIP

new Quaternion();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();
new Vec3();

new AABB();
new Ray(); // performance.now() fallback on Date.now()

const performance = globalThis.performance || {};

if (!performance.now) {
  let nowOffset = Date.now();

  if (performance.timing && performance.timing.navigationStart) {
    nowOffset = performance.timing.navigationStart;
  }

  performance.now = () => Date.now() - nowOffset;
}

new Vec3(); // Dispatched after the world has stepped forward in time.
({
  type: Body.COLLIDE_EVENT_NAME,
  body: null,
  contact: null
}); // Pools for unused objects

function WorkerWrapper() {
          return new Worker(new URL("./physicsworker.733825cf.js", import.meta.url), {
  "type": "module"
})
        }

const PhysicsData = Body;
const ConstraintData = PointToPointConstraint;
let log = console.log;
let report = console.error;
class Physics {
  #worker;
  #idCounter = 0;
  #raycastIdCounter = 0;
  #tbuffer = new SharedArrayBuffer(4 * 16 * 1024);
  #tview = new Float32Array(this.#tbuffer);
  #idToBody = /* @__PURE__ */ new Map();
  #bodyToId = /* @__PURE__ */ new WeakMap();
  #idToEntity = /* @__PURE__ */ new Map();
  #collisionCallbacks = /* @__PURE__ */ new Map();
  #raycastCallbacks = /* @__PURE__ */ new Map();
  constructor() {
    this.#worker = new WorkerWrapper();
  }
  async init(events, logService) {
    if (logService)
      [log, report] = logService;
    log(`Starting
${import.meta.url}`);
    events.on(`set${PhysicsData.name}Component`, (entityId, body) => {
      const bodyId = this.#bodyToId.get(body);
      this.#idToEntity.set(bodyId, entityId);
    });
    events.on(`delete${PhysicsData.name}Component`, (_, body) => {
      this.removeBody(body);
    });
    return new Promise((resolve) => {
      this.#worker.onerror = report;
      this.#worker.onmessage = ({ data }) => {
        if (data === "frogbones")
          log("ayye");
        switch (data.type) {
          case "ready": {
            log("\u2515 Ready");
            this.#worker.postMessage({ type: "init", buffer: this.#tbuffer });
            resolve();
            break;
          }
          case "collisions": {
            const { collisions } = data;
            for (let i = 0; i < collisions.length; i += 2) {
              const rbId0 = collisions[i + 0];
              const rbId1 = collisions[i + 1];
              const id0 = this.#idToEntity.get(rbId0);
              const id1 = this.#idToEntity.get(rbId1);
              events.emit("collision", { id0, id1 });
              this.#collisionCallbacks.get(rbId0)?.(id1);
            }
            break;
          }
          case "raycastResult": {
            const { raycastId, bodyId, hitPoint } = data;
            const entityID = this.#idToEntity.get(bodyId);
            const { x, y, z } = hitPoint;
            this.#raycastCallbacks.get(raycastId)({
              entityID,
              hitPoint: new Vec3(x, y, z)
            });
            break;
          }
          default: {
            throw new Error(`[physics] unknown message type ${data.type}`);
          }
        }
      };
    });
  }
  update() {
    for (const [id, body] of this.#idToBody) {
      const offset = 3 * id;
      body.position.x = this.#tview[offset + 0];
      body.position.y = this.#tview[offset + 1];
      body.position.z = this.#tview[offset + 2];
    }
  }
  registerCollisionCallback(body, cb) {
    const id = this.#bodyToId.get(body);
    this.#collisionCallbacks.set(id, cb);
  }
  removeCollisionCallback(body) {
    const id = this.#bodyToId.get(body);
    this.#collisionCallbacks.delete(id);
  }
  addForce(body, force) {
    this.#worker.postMessage({
      type: "addForce",
      id: this.#bodyToId.get(body),
      x: force.x,
      y: force.y,
      z: force.z
    });
  }
  addForceConditionalRaycast(body, force, from, to) {
    this.#worker.postMessage({
      type: "addForceConditionalRaycast",
      id: this.#bodyToId.get(body),
      x: force.x,
      y: force.y,
      z: force.z,
      fx: from.x,
      fy: from.y,
      fz: from.z,
      tx: to.x,
      ty: to.y,
      tz: to.z
    });
  }
  addVelocity(body, velocity) {
    this.#worker.postMessage({
      type: "addVelocity",
      id: this.#bodyToId.get(body),
      x: velocity.x,
      y: velocity.y,
      z: velocity.z
    });
  }
  addVelocityConditionalRaycast(body, velocity, from, to) {
    this.#worker.postMessage({
      type: "addVelocityConditionalRaycast",
      id: this.#bodyToId.get(body),
      vx: velocity.x,
      vy: velocity.y,
      vz: velocity.z,
      fx: from.x,
      fy: from.y,
      fz: from.z,
      tx: to.x,
      ty: to.y,
      tz: to.z
    });
  }
  raycast(from, to) {
    return new Promise((resolve) => {
      const id = this.#raycastIdCounter;
      this.#raycastIdCounter += 1;
      this.#raycastCallbacks.set(id, resolve);
      this.#worker.postMessage({
        type: "raycast",
        id,
        fx: from.x,
        fy: from.y,
        fz: from.z,
        tx: to.x,
        ty: to.y,
        tz: to.z
      });
    });
  }
  removeBody(body) {
    const id = this.#bodyToId.get(body);
    this.#worker.postMessage({
      type: "removeBody",
      id
    });
    this.#bodyToId.delete(body);
    this.#idToBody.delete(id);
  }
  createTrimesh(opts, geometry) {
    const id = this.#idCounter;
    this.#idCounter += 1;
    const nonIndexedGeo = geometry.index ? geometry.toNonIndexed() : geometry;
    const triangles = nonIndexedGeo.getAttribute("position").array;
    const triangleBuffer = triangles.buffer;
    this.#worker.postMessage({
      type: "createTrimesh",
      triangleBuffer,
      x: opts.pos?.x ?? 0,
      y: opts.pos?.y ?? 0,
      z: opts.pos?.z ?? 0,
      sx: opts.scale?.x ?? 1,
      sy: opts.scale?.y ?? 1,
      sz: opts.scale?.z ?? 1,
      qx: opts.quat?.x ?? 0,
      qy: opts.quat?.y ?? 0,
      qz: opts.quat?.z ?? 0,
      qw: opts.quat?.w ?? 1,
      id
    }, []);
    const mock = new Body();
    this.#idToBody.set(id, mock);
    this.#bodyToId.set(mock, id);
    return mock;
  }
  createSphere(opts, radius) {
    const id = this.#idCounter;
    this.#idCounter += 1;
    this.#worker.postMessage({
      type: "createSphere",
      radius,
      mass: opts.mass,
      x: opts.pos?.x ?? 0,
      y: opts.pos?.y ?? 0,
      z: opts.pos?.z ?? 0,
      sx: opts.scale?.x ?? 1,
      sy: opts.scale?.y ?? 1,
      sz: opts.scale?.z ?? 1,
      qx: opts.quat?.x ?? 0,
      qy: opts.quat?.y ?? 0,
      qz: opts.quat?.z ?? 0,
      qw: opts.quat?.w ?? 1,
      fixedRotation: opts.fixedRotation ?? false,
      id
    });
    const mock = new Body();
    this.#idToBody.set(id, mock);
    this.#bodyToId.set(mock, id);
    return mock;
  }
  createCapsule(opts, radius, height) {
    const id = this.#idCounter;
    this.#idCounter += 1;
    this.#worker.postMessage({
      type: "createCapsule",
      radius,
      height,
      mass: opts.mass,
      x: opts.pos?.x ?? 0,
      y: opts.pos?.y ?? 0,
      z: opts.pos?.z ?? 0,
      sx: opts.scale?.x ?? 1,
      sy: opts.scale?.y ?? 1,
      sz: opts.scale?.z ?? 1,
      qx: opts.quat?.x ?? 0,
      qy: opts.quat?.y ?? 0,
      qz: opts.quat?.z ?? 0,
      qw: opts.quat?.w ?? 1,
      fixedRotation: opts.fixedRotation ?? false,
      id
    });
    const mock = new Body();
    this.#idToBody.set(id, mock);
    this.#bodyToId.set(mock, id);
    return mock;
  }
  static makeCube(mass, size) {
    const shape = new Box(new Vec3(size, size, size));
    const body = new Body({ mass });
    body.addShape(shape);
    return body;
  }
  static makeBall(mass, radius) {
    const shape = new Sphere(radius);
    const body = new Body({ mass });
    body.addShape(shape);
    return body;
  }
  static makeCylinder(mass, radius, height) {
    const shape = new Cylinder(radius, radius, height);
    const body = new Body({ mass });
    body.addShape(shape);
    return body;
  }
  static makeCapsule(mass, radius, height) {
    const shape = new Sphere(radius);
    const body = new Body({ mass });
    body.addShape(shape, new Vec3(0, 0, 0));
    body.addShape(shape, new Vec3(0, height / 2 - radius, 0));
    body.addShape(shape, new Vec3(0, -height / 2 + radius, 0));
    return body;
  }
}

export { ConstraintData, Physics, PhysicsData };
