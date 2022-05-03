var __accessCheck = (obj, member, msg) => {
  if (!member.has(obj))
    throw TypeError("Cannot " + msg);
};
var __privateGet = (obj, member, getter) => {
  __accessCheck(obj, member, "read from private field");
  return getter ? getter.call(obj) : member.get(obj);
};
var __privateAdd = (obj, member, value) => {
  if (member.has(obj))
    throw TypeError("Cannot add the same private member more than once");
  member instanceof WeakSet ? member.add(obj) : member.set(obj, value);
};
var __privateSet = (obj, member, value, setter) => {
  __accessCheck(obj, member, "write to private field");
  setter ? setter.call(obj, value) : member.set(obj, value);
  return value;
};
var _worker, _idCounter, _raycastIdCounter, _tbuffer, _tview, _idToBody, _bodyToId, _idToEntity, _collisionCallbacks, _raycastCallbacks;
class Mat3 {
  constructor(elements) {
    if (elements === void 0) {
      elements = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    this.elements = elements;
  }
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
  setTrace(vector) {
    const e = this.elements;
    e[0] = vector.x;
    e[4] = vector.y;
    e[8] = vector.z;
  }
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
  smult(s) {
    for (let i = 0; i < this.elements.length; i++) {
      this.elements[i] *= s;
    }
  }
  mmult(matrix, target) {
    if (target === void 0) {
      target = new Mat3();
    }
    const A = this.elements;
    const B = matrix.elements;
    const T = target.elements;
    const a11 = A[0], a12 = A[1], a13 = A[2], a21 = A[3], a22 = A[4], a23 = A[5], a31 = A[6], a32 = A[7], a33 = A[8];
    const b11 = B[0], b12 = B[1], b13 = B[2], b21 = B[3], b22 = B[4], b23 = B[5], b31 = B[6], b32 = B[7], b33 = B[8];
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
  solve(b2, target) {
    if (target === void 0) {
      target = new Vec3();
    }
    const nr = 3;
    const nc = 4;
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
    eqns[3 + 4 * 0] = b2.x;
    eqns[3 + 4 * 1] = b2.y;
    eqns[3 + 4 * 2] = b2.z;
    let n = 3;
    const k = n;
    let np;
    const kp = 4;
    let p;
    do {
      i = k - n;
      if (eqns[i + nc * i] === 0) {
        for (j = i + 1; j < k; j++) {
          if (eqns[i + nc * j] !== 0) {
            np = kp;
            do {
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
            p = kp - np;
            eqns[p + nc * j] = p <= i ? 0 : eqns[p + nc * j] - eqns[p + nc * i] * multiplier;
          } while (--np);
        }
      }
    } while (--n);
    target.z = eqns[2 * nc + 3] / eqns[2 * nc + 2];
    target.y = (eqns[1 * nc + 3] - eqns[1 * nc + 2] * target.z) / eqns[1 * nc + 1];
    target.x = (eqns[0 * nc + 3] - eqns[0 * nc + 2] * target.z - eqns[0 * nc + 1] * target.y) / eqns[0 * nc + 0];
    if (isNaN(target.x) || isNaN(target.y) || isNaN(target.z) || target.x === Infinity || target.y === Infinity || target.z === Infinity) {
      throw `Could not solve equation! Got x=[${target.toString()}], b=[${b2.toString()}], A=[${this.toString()}]`;
    }
    return target;
  }
  e(row, column, value) {
    if (value === void 0) {
      return this.elements[column + 3 * row];
    } else {
      this.elements[column + 3 * row] = value;
    }
  }
  copy(matrix) {
    for (let i = 0; i < matrix.elements.length; i++) {
      this.elements[i] = matrix.elements[i];
    }
    return this;
  }
  toString() {
    let r = "";
    const sep = ",";
    for (let i = 0; i < 9; i++) {
      r += this.elements[i] + sep;
    }
    return r;
  }
  reverse(target) {
    if (target === void 0) {
      target = new Mat3();
    }
    const nr = 3;
    const nc = 6;
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
    eqns[5 + 6 * 2] = 1;
    let n = 3;
    const k = n;
    let np;
    const kp = nc;
    let p;
    do {
      i = k - n;
      if (eqns[i + nc * i] === 0) {
        for (j = i + 1; j < k; j++) {
          if (eqns[i + nc * j] !== 0) {
            np = kp;
            do {
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
            p = kp - np;
            eqns[p + nc * j] = p <= i ? 0 : eqns[p + nc * j] - eqns[p + nc * i] * multiplier;
          } while (--np);
        }
      }
    } while (--n);
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
    } while (--i);
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
  transpose(target) {
    if (target === void 0) {
      target = new Mat3();
    }
    const M = this.elements;
    const T = target.elements;
    let tmp2;
    T[0] = M[0];
    T[4] = M[4];
    T[8] = M[8];
    tmp2 = M[1];
    T[1] = M[3];
    T[3] = tmp2;
    tmp2 = M[2];
    T[2] = M[6];
    T[6] = tmp2;
    tmp2 = M[5];
    T[5] = M[7];
    T[7] = tmp2;
    return target;
  }
}
const reverse_eqns = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
class Vec3 {
  constructor(x, y, z) {
    if (x === void 0) {
      x = 0;
    }
    if (y === void 0) {
      y = 0;
    }
    if (z === void 0) {
      z = 0;
    }
    this.x = x;
    this.y = y;
    this.z = z;
  }
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
  set(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  setZero() {
    this.x = this.y = this.z = 0;
  }
  vadd(vector, target) {
    if (target) {
      target.x = vector.x + this.x;
      target.y = vector.y + this.y;
      target.z = vector.z + this.z;
    } else {
      return new Vec3(this.x + vector.x, this.y + vector.y, this.z + vector.z);
    }
  }
  vsub(vector, target) {
    if (target) {
      target.x = this.x - vector.x;
      target.y = this.y - vector.y;
      target.z = this.z - vector.z;
    } else {
      return new Vec3(this.x - vector.x, this.y - vector.y, this.z - vector.z);
    }
  }
  crossmat() {
    return new Mat3([0, -this.z, this.y, this.z, 0, -this.x, -this.y, this.x, 0]);
  }
  normalize() {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const n = Math.sqrt(x * x + y * y + z * z);
    if (n > 0) {
      const invN = 1 / n;
      this.x *= invN;
      this.y *= invN;
      this.z *= invN;
    } else {
      this.x = 0;
      this.y = 0;
      this.z = 0;
    }
    return n;
  }
  unit(target) {
    if (target === void 0) {
      target = new Vec3();
    }
    const x = this.x;
    const y = this.y;
    const z = this.z;
    let ninv = Math.sqrt(x * x + y * y + z * z);
    if (ninv > 0) {
      ninv = 1 / ninv;
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
  length() {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    return Math.sqrt(x * x + y * y + z * z);
  }
  lengthSquared() {
    return this.dot(this);
  }
  distanceTo(p) {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const px = p.x;
    const py = p.y;
    const pz = p.z;
    return Math.sqrt((px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z));
  }
  distanceSquared(p) {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const px = p.x;
    const py = p.y;
    const pz = p.z;
    return (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z);
  }
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
  vmul(vector, target) {
    if (target === void 0) {
      target = new Vec3();
    }
    target.x = vector.x * this.x;
    target.y = vector.y * this.y;
    target.z = vector.z * this.z;
    return target;
  }
  addScaledVector(scalar, vector, target) {
    if (target === void 0) {
      target = new Vec3();
    }
    target.x = this.x + scalar * vector.x;
    target.y = this.y + scalar * vector.y;
    target.z = this.z + scalar * vector.z;
    return target;
  }
  dot(vector) {
    return this.x * vector.x + this.y * vector.y + this.z * vector.z;
  }
  isZero() {
    return this.x === 0 && this.y === 0 && this.z === 0;
  }
  negate(target) {
    if (target === void 0) {
      target = new Vec3();
    }
    target.x = -this.x;
    target.y = -this.y;
    target.z = -this.z;
    return target;
  }
  tangents(t1, t2) {
    const norm = this.length();
    if (norm > 0) {
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
      t1.set(1, 0, 0);
      t2.set(0, 1, 0);
    }
  }
  toString() {
    return `${this.x},${this.y},${this.z}`;
  }
  toArray() {
    return [this.x, this.y, this.z];
  }
  copy(vector) {
    this.x = vector.x;
    this.y = vector.y;
    this.z = vector.z;
    return this;
  }
  lerp(vector, t, target) {
    const x = this.x;
    const y = this.y;
    const z = this.z;
    target.x = x + (vector.x - x) * t;
    target.y = y + (vector.y - y) * t;
    target.z = z + (vector.z - z) * t;
  }
  almostEquals(vector, precision) {
    if (precision === void 0) {
      precision = 1e-6;
    }
    if (Math.abs(this.x - vector.x) > precision || Math.abs(this.y - vector.y) > precision || Math.abs(this.z - vector.z) > precision) {
      return false;
    }
    return true;
  }
  almostZero(precision) {
    if (precision === void 0) {
      precision = 1e-6;
    }
    if (Math.abs(this.x) > precision || Math.abs(this.y) > precision || Math.abs(this.z) > precision) {
      return false;
    }
    return true;
  }
  isAntiparallelTo(vector, precision) {
    this.negate(antip_neg);
    return antip_neg.almostEquals(vector, precision);
  }
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
class AABB {
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
  setFromPoints(points, position, quaternion, skinSize) {
    const l = this.lowerBound;
    const u = this.upperBound;
    const q = quaternion;
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
    }
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
  copy(aabb) {
    this.lowerBound.copy(aabb.lowerBound);
    this.upperBound.copy(aabb.upperBound);
    return this;
  }
  clone() {
    return new AABB().copy(this);
  }
  extend(aabb) {
    this.lowerBound.x = Math.min(this.lowerBound.x, aabb.lowerBound.x);
    this.upperBound.x = Math.max(this.upperBound.x, aabb.upperBound.x);
    this.lowerBound.y = Math.min(this.lowerBound.y, aabb.lowerBound.y);
    this.upperBound.y = Math.max(this.upperBound.y, aabb.upperBound.y);
    this.lowerBound.z = Math.min(this.lowerBound.z, aabb.lowerBound.z);
    this.upperBound.z = Math.max(this.upperBound.z, aabb.upperBound.z);
  }
  overlaps(aabb) {
    const l1 = this.lowerBound;
    const u1 = this.upperBound;
    const l2 = aabb.lowerBound;
    const u2 = aabb.upperBound;
    const overlapsX = l2.x <= u1.x && u1.x <= u2.x || l1.x <= u2.x && u2.x <= u1.x;
    const overlapsY = l2.y <= u1.y && u1.y <= u2.y || l1.y <= u2.y && u2.y <= u1.y;
    const overlapsZ = l2.z <= u1.z && u1.z <= u2.z || l1.z <= u2.z && u2.z <= u1.z;
    return overlapsX && overlapsY && overlapsZ;
  }
  volume() {
    const l = this.lowerBound;
    const u = this.upperBound;
    return (u.x - l.x) * (u.y - l.y) * (u.z - l.z);
  }
  contains(aabb) {
    const l1 = this.lowerBound;
    const u1 = this.upperBound;
    const l2 = aabb.lowerBound;
    const u2 = aabb.upperBound;
    return l1.x <= l2.x && u1.x >= u2.x && l1.y <= l2.y && u1.y >= u2.y && l1.z <= l2.z && u1.z >= u2.z;
  }
  getCorners(a2, b2, c2, d, e, f, g, h) {
    const l = this.lowerBound;
    const u = this.upperBound;
    a2.copy(l);
    b2.set(u.x, l.y, l.z);
    c2.set(u.x, u.y, l.z);
    d.set(l.x, u.y, u.z);
    e.set(u.x, l.y, u.z);
    f.set(l.x, u.y, l.z);
    g.set(l.x, l.y, u.z);
    h.copy(u);
  }
  toLocalFrame(frame, target) {
    const corners = transformIntoFrame_corners;
    const a2 = corners[0];
    const b2 = corners[1];
    const c2 = corners[2];
    const d = corners[3];
    const e = corners[4];
    const f = corners[5];
    const g = corners[6];
    const h = corners[7];
    this.getCorners(a2, b2, c2, d, e, f, g, h);
    for (let i = 0; i !== 8; i++) {
      const corner = corners[i];
      frame.pointToLocal(corner, corner);
    }
    return target.setFromPoints(corners);
  }
  toWorldFrame(frame, target) {
    const corners = transformIntoFrame_corners;
    const a2 = corners[0];
    const b2 = corners[1];
    const c2 = corners[2];
    const d = corners[3];
    const e = corners[4];
    const f = corners[5];
    const g = corners[6];
    const h = corners[7];
    this.getCorners(a2, b2, c2, d, e, f, g, h);
    for (let i = 0; i !== 8; i++) {
      const corner = corners[i];
      frame.pointToWorld(corner, corner);
    }
    return target.setFromPoints(corners);
  }
  overlapsRay(ray) {
    const {
      direction,
      from
    } = ray;
    const dirFracX = 1 / direction.x;
    const dirFracY = 1 / direction.y;
    const dirFracZ = 1 / direction.z;
    const t1 = (this.lowerBound.x - from.x) * dirFracX;
    const t2 = (this.upperBound.x - from.x) * dirFracX;
    const t3 = (this.lowerBound.y - from.y) * dirFracY;
    const t4 = (this.upperBound.y - from.y) * dirFracY;
    const t5 = (this.lowerBound.z - from.z) * dirFracZ;
    const t6 = (this.upperBound.z - from.z) * dirFracZ;
    const tmin = Math.max(Math.max(Math.min(t1, t2), Math.min(t3, t4)), Math.min(t5, t6));
    const tmax = Math.min(Math.min(Math.max(t1, t2), Math.max(t3, t4)), Math.max(t5, t6));
    if (tmax < 0) {
      return false;
    }
    if (tmin > tmax) {
      return false;
    }
    return true;
  }
}
const tmp$1 = new Vec3();
const transformIntoFrame_corners = [new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3()];
class EventTarget {
  addEventListener(type, listener) {
    if (this._listeners === void 0) {
      this._listeners = {};
    }
    const listeners = this._listeners;
    if (listeners[type] === void 0) {
      listeners[type] = [];
    }
    if (!listeners[type].includes(listener)) {
      listeners[type].push(listener);
    }
    return this;
  }
  hasEventListener(type, listener) {
    if (this._listeners === void 0) {
      return false;
    }
    const listeners = this._listeners;
    if (listeners[type] !== void 0 && listeners[type].includes(listener)) {
      return true;
    }
    return false;
  }
  hasAnyEventListener(type) {
    if (this._listeners === void 0) {
      return false;
    }
    const listeners = this._listeners;
    return listeners[type] !== void 0;
  }
  removeEventListener(type, listener) {
    if (this._listeners === void 0) {
      return this;
    }
    const listeners = this._listeners;
    if (listeners[type] === void 0) {
      return this;
    }
    const index = listeners[type].indexOf(listener);
    if (index !== -1) {
      listeners[type].splice(index, 1);
    }
    return this;
  }
  dispatchEvent(event) {
    if (this._listeners === void 0) {
      return this;
    }
    const listeners = this._listeners;
    const listenerArray = listeners[event.type];
    if (listenerArray !== void 0) {
      event.target = this;
      for (let i = 0, l = listenerArray.length; i < l; i++) {
        listenerArray[i].call(this, event);
      }
    }
    return this;
  }
}
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
  set(x, y, z, w) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    return this;
  }
  toString() {
    return `${this.x},${this.y},${this.z},${this.w}`;
  }
  toArray() {
    return [this.x, this.y, this.z, this.w];
  }
  setFromAxisAngle(vector, angle) {
    const s = Math.sin(angle * 0.5);
    this.x = vector.x * s;
    this.y = vector.y * s;
    this.z = vector.z * s;
    this.w = Math.cos(angle * 0.5);
    return this;
  }
  toAxisAngle(targetAxis) {
    if (targetAxis === void 0) {
      targetAxis = new Vec3();
    }
    this.normalize();
    const angle = 2 * Math.acos(this.w);
    const s = Math.sqrt(1 - this.w * this.w);
    if (s < 1e-3) {
      targetAxis.x = this.x;
      targetAxis.y = this.y;
      targetAxis.z = this.z;
    } else {
      targetAxis.x = this.x / s;
      targetAxis.y = this.y / s;
      targetAxis.z = this.z / s;
    }
    return [targetAxis, angle];
  }
  setFromVectors(u, v) {
    if (u.isAntiparallelTo(v)) {
      const t1 = sfv_t1;
      const t2 = sfv_t2;
      u.tangents(t1, t2);
      this.setFromAxisAngle(t1, Math.PI);
    } else {
      const a2 = u.cross(v);
      this.x = a2.x;
      this.y = a2.y;
      this.z = a2.z;
      this.w = Math.sqrt(u.length() ** 2 * v.length() ** 2) + u.dot(v);
      this.normalize();
    }
    return this;
  }
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
  normalizeFast() {
    const f = (3 - (this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w)) / 2;
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
    const qw = this.w;
    const ix = qw * x + qy * z - qz * y;
    const iy = qw * y + qz * x - qx * z;
    const iz = qw * z + qx * y - qy * x;
    const iw = -qx * x - qy * y - qz * z;
    target.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    target.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    target.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;
    return target;
  }
  copy(quat) {
    this.x = quat.x;
    this.y = quat.y;
    this.z = quat.z;
    this.w = quat.w;
    return this;
  }
  toEuler(target, order) {
    if (order === void 0) {
      order = "YZX";
    }
    let heading;
    let attitude;
    let bank;
    const x = this.x;
    const y = this.y;
    const z = this.z;
    const w = this.w;
    switch (order) {
      case "YZX":
        const test = x * y + z * w;
        if (test > 0.499) {
          heading = 2 * Math.atan2(x, w);
          attitude = Math.PI / 2;
          bank = 0;
        }
        if (test < -0.499) {
          heading = -2 * Math.atan2(x, w);
          attitude = -Math.PI / 2;
          bank = 0;
        }
        if (heading === void 0) {
          const sqx = x * x;
          const sqy = y * y;
          const sqz = z * z;
          heading = Math.atan2(2 * y * w - 2 * x * z, 1 - 2 * sqy - 2 * sqz);
          attitude = Math.asin(2 * test);
          bank = Math.atan2(2 * x * w - 2 * y * z, 1 - 2 * sqx - 2 * sqz);
        }
        break;
      default:
        throw new Error(`Euler order ${order} not supported yet.`);
    }
    target.y = heading;
    target.z = attitude;
    target.x = bank;
  }
  setFromEuler(x, y, z, order) {
    if (order === void 0) {
      order = "XYZ";
    }
    const c1 = Math.cos(x / 2);
    const c2 = Math.cos(y / 2);
    const c3 = Math.cos(z / 2);
    const s1 = Math.sin(x / 2);
    const s2 = Math.sin(y / 2);
    const s3 = Math.sin(z / 2);
    if (order === "XYZ") {
      this.x = s1 * c2 * c3 + c1 * s2 * s3;
      this.y = c1 * s2 * c3 - s1 * c2 * s3;
      this.z = c1 * c2 * s3 + s1 * s2 * c3;
      this.w = c1 * c2 * c3 - s1 * s2 * s3;
    } else if (order === "YXZ") {
      this.x = s1 * c2 * c3 + c1 * s2 * s3;
      this.y = c1 * s2 * c3 - s1 * c2 * s3;
      this.z = c1 * c2 * s3 - s1 * s2 * c3;
      this.w = c1 * c2 * c3 + s1 * s2 * s3;
    } else if (order === "ZXY") {
      this.x = s1 * c2 * c3 - c1 * s2 * s3;
      this.y = c1 * s2 * c3 + s1 * c2 * s3;
      this.z = c1 * c2 * s3 + s1 * s2 * c3;
      this.w = c1 * c2 * c3 - s1 * s2 * s3;
    } else if (order === "ZYX") {
      this.x = s1 * c2 * c3 - c1 * s2 * s3;
      this.y = c1 * s2 * c3 + s1 * c2 * s3;
      this.z = c1 * c2 * s3 - s1 * s2 * c3;
      this.w = c1 * c2 * c3 + s1 * s2 * s3;
    } else if (order === "YZX") {
      this.x = s1 * c2 * c3 + c1 * s2 * s3;
      this.y = c1 * s2 * c3 + s1 * c2 * s3;
      this.z = c1 * c2 * s3 - s1 * s2 * c3;
      this.w = c1 * c2 * c3 - s1 * s2 * s3;
    } else if (order === "XZY") {
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
    let scale1;
    cosom = ax * bx + ay * by + az * bz + aw * bw;
    if (cosom < 0) {
      cosom = -cosom;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    if (1 - cosom > 1e-6) {
      omega = Math.acos(cosom);
      sinom = Math.sin(omega);
      scale0 = Math.sin((1 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      scale0 = 1 - t;
      scale1 = t;
    }
    target.x = scale0 * ax + scale1 * bx;
    target.y = scale0 * ay + scale1 * by;
    target.z = scale0 * az + scale1 * bz;
    target.w = scale0 * aw + scale1 * bw;
    return target;
  }
  integrate(angularVelocity, dt, angularFactor, target) {
    if (target === void 0) {
      target = new Quaternion();
    }
    const ax = angularVelocity.x * angularFactor.x, ay = angularVelocity.y * angularFactor.y, az = angularVelocity.z * angularFactor.z, bx = this.x, by = this.y, bz = this.z, bw = this.w;
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
const SHAPE_TYPES = {
  SPHERE: 1,
  PLANE: 2,
  BOX: 4,
  COMPOUND: 8,
  CONVEXPOLYHEDRON: 16,
  HEIGHTFIELD: 32,
  PARTICLE: 64,
  CYLINDER: 128,
  TRIMESH: 256
};
class Shape {
  constructor(options) {
    if (options === void 0) {
      options = {};
    }
    this.id = Shape.idCounter++;
    this.type = options.type || 0;
    this.boundingSphereRadius = 0;
    this.collisionResponse = options.collisionResponse ? options.collisionResponse : true;
    this.collisionFilterGroup = options.collisionFilterGroup !== void 0 ? options.collisionFilterGroup : 1;
    this.collisionFilterMask = options.collisionFilterMask !== void 0 ? options.collisionFilterMask : -1;
    this.material = options.material ? options.material : null;
    this.body = null;
  }
  updateBoundingSphereRadius() {
    throw `computeBoundingSphereRadius() not implemented for shape type ${this.type}`;
  }
  volume() {
    throw `volume() not implemented for shape type ${this.type}`;
  }
  calculateLocalInertia(mass, target) {
    throw `calculateLocalInertia() not implemented for shape type ${this.type}`;
  }
  calculateWorldAABB(pos, quat, min, max) {
    throw `calculateWorldAABB() not implemented for shape type ${this.type}`;
  }
}
Shape.idCounter = 0;
Shape.types = SHAPE_TYPES;
class Transform {
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
  pointToLocal(worldPoint, result) {
    return Transform.pointToLocalFrame(this.position, this.quaternion, worldPoint, result);
  }
  pointToWorld(localPoint, result) {
    return Transform.pointToWorldFrame(this.position, this.quaternion, localPoint, result);
  }
  vectorToWorldFrame(localVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    this.quaternion.vmult(localVector, result);
    return result;
  }
  static pointToLocalFrame(position, quaternion, worldPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    worldPoint.vsub(position, result);
    quaternion.conjugate(tmpQuat$1);
    tmpQuat$1.vmult(result, result);
    return result;
  }
  static pointToWorldFrame(position, quaternion, localPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    quaternion.vmult(localPoint, result);
    result.vadd(position, result);
    return result;
  }
  static vectorToWorldFrame(quaternion, localVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    quaternion.vmult(localVector, result);
    return result;
  }
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
class ConvexPolyhedron extends Shape {
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
    this.worldVertices = [];
    this.worldVerticesNeedsUpdate = true;
    this.worldFaceNormals = [];
    this.worldFaceNormalsNeedsUpdate = true;
    this.uniqueAxes = axes ? axes.slice() : null;
    this.uniqueEdges = [];
    this.computeEdges();
  }
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
  computeNormals() {
    this.faceNormals.length = this.faces.length;
    for (let i = 0; i < this.faces.length; i++) {
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
  getFaceNormal(i, target) {
    const f = this.faces[i];
    const va = this.vertices[f[0]];
    const vb = this.vertices[f[1]];
    const vc = this.vertices[f[2]];
    ConvexPolyhedron.computeNormal(va, vb, vc, target);
  }
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
      const b2 = hullB.vertices[hullB.faces[closestFaceB][i]];
      const worldb = new Vec3();
      worldb.copy(b2);
      quatB.vmult(worldb, worldb);
      posB.vadd(worldb, worldb);
      worldVertsB1.push(worldb);
    }
    if (closestFaceB >= 0) {
      this.clipFaceAgainstHull(separatingNormal, posA, quatA, worldVertsB1, minDist, maxDist, result);
    }
  }
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
      const numFacesA = faceListA ? faceListA.length : hullA.faces.length;
      for (let i = 0; i < numFacesA; i++) {
        const fi = faceListA ? faceListA[i] : i;
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
      for (let i = 0; i !== hullA.uniqueAxes.length; i++) {
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
    }
    for (let e0 = 0; e0 !== hullA.uniqueEdges.length; e0++) {
      quatA.vmult(hullA.uniqueEdges[e0], worldEdge0);
      for (let e1 = 0; e1 !== hullB.uniqueEdges.length; e1++) {
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
    if (deltaC.dot(target) > 0) {
      target.negate(target);
    }
    return true;
  }
  testSepAxis(axis, hullB, posA, quatA, posB, quatB) {
    const hullA = this;
    ConvexPolyhedron.project(hullA, axis, posA, quatA, maxminA);
    ConvexPolyhedron.project(hullB, axis, posB, quatB, maxminB);
    const maxA = maxminA[0];
    const minA = maxminA[1];
    const maxB = maxminB[0];
    const minB = maxminB[1];
    if (maxA < minB || maxB < minA) {
      return false;
    }
    const d0 = maxA - minB;
    const d1 = maxB - minA;
    const depth = d0 < d1 ? d0 : d1;
    return depth;
  }
  calculateLocalInertia(mass, target) {
    const aabbmax = new Vec3();
    const aabbmin = new Vec3();
    this.computeLocalAABB(aabbmin, aabbmax);
    const x = aabbmax.x - aabbmin.x;
    const y = aabbmax.y - aabbmin.y;
    const z = aabbmax.z - aabbmin.z;
    target.x = 1 / 12 * mass * (2 * y * 2 * y + 2 * z * 2 * z);
    target.y = 1 / 12 * mass * (2 * x * 2 * x + 2 * z * 2 * z);
    target.z = 1 / 12 * mass * (2 * y * 2 * y + 2 * x * 2 * x);
  }
  getPlaneConstantOfFace(face_i) {
    const f = this.faces[face_i];
    const n = this.faceNormals[face_i];
    const v = this.vertices[f[0]];
    const c2 = -n.dot(v);
    return c2;
  }
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
    let dmin = Number.MAX_VALUE;
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
    }
    const polyA = hullA.faces[closestFaceA];
    polyA.connectedFaces = [];
    for (let i = 0; i < hullA.faces.length; i++) {
      for (let j = 0; j < hullA.faces[i].length; j++) {
        if (polyA.indexOf(hullA.faces[i][j]) !== -1 && i !== closestFaceA && polyA.connectedFaces.indexOf(i) === -1) {
          polyA.connectedFaces.push(i);
        }
      }
    }
    const numVerticesA = polyA.length;
    for (let i = 0; i < numVerticesA; i++) {
      const a2 = hullA.vertices[polyA[i]];
      const b2 = hullA.vertices[polyA[(i + 1) % numVerticesA]];
      a2.vsub(b2, edge0);
      WorldEdge0.copy(edge0);
      quatA.vmult(WorldEdge0, WorldEdge0);
      posA.vadd(WorldEdge0, WorldEdge0);
      worldPlaneAnormal1.copy(this.faceNormals[closestFaceA]);
      quatA.vmult(worldPlaneAnormal1, worldPlaneAnormal1);
      posA.vadd(worldPlaneAnormal1, worldPlaneAnormal1);
      WorldEdge0.cross(worldPlaneAnormal1, planeNormalWS1);
      planeNormalWS1.negate(planeNormalWS1);
      worldA1.copy(a2);
      quatA.vmult(worldA1, worldA1);
      posA.vadd(worldA1, worldA1);
      const otherFace = polyA.connectedFaces[i];
      localPlaneNormal.copy(this.faceNormals[otherFace]);
      const localPlaneEq2 = this.getPlaneConstantOfFace(otherFace);
      planeNormalWS.copy(localPlaneNormal);
      quatA.vmult(planeNormalWS, planeNormalWS);
      const planeEqWS2 = localPlaneEq2 - planeNormalWS.dot(posA);
      this.clipFaceAgainstPlane(pVtxIn, pVtxOut, planeNormalWS, planeEqWS2);
      while (pVtxIn.length) {
        pVtxIn.shift();
      }
      while (pVtxOut.length) {
        pVtxIn.push(pVtxOut.shift());
      }
    }
    localPlaneNormal.copy(this.faceNormals[closestFaceA]);
    const localPlaneEq = this.getPlaneConstantOfFace(closestFaceA);
    planeNormalWS.copy(localPlaneNormal);
    quatA.vmult(planeNormalWS, planeNormalWS);
    const planeEqWS = localPlaneEq - planeNormalWS.dot(posA);
    for (let i = 0; i < pVtxIn.length; i++) {
      let depth = planeNormalWS.dot(pVtxIn[i]) + planeEqWS;
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
          const newv = new Vec3();
          newv.copy(lastVertex);
          outVertices.push(newv);
        } else {
          const newv = new Vec3();
          firstVertex.lerp(lastVertex, n_dot_first / (n_dot_first - n_dot_last), newv);
          outVertices.push(newv);
        }
      } else {
        if (n_dot_last < 0) {
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
  updateBoundingSphereRadius() {
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
      if (minx === void 0 || v.x < minx) {
        minx = v.x;
      }
      if (maxx === void 0 || v.x > maxx) {
        maxx = v.x;
      }
      if (miny === void 0 || v.y < miny) {
        miny = v.y;
      }
      if (maxy === void 0 || v.y > maxy) {
        maxy = v.y;
      }
      if (minz === void 0 || v.z < minz) {
        minz = v.z;
      }
      if (maxz === void 0 || v.z > maxz) {
        maxz = v.z;
      }
    }
    min.set(minx, miny, minz);
    max.set(maxx, maxy, maxz);
  }
  volume() {
    return 4 * Math.PI * this.boundingSphereRadius / 3;
  }
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
  transformAllPoints(offset, quat) {
    const n = this.vertices.length;
    const verts = this.vertices;
    if (quat) {
      for (let i = 0; i < n; i++) {
        const v = verts[i];
        quat.vmult(v, v);
      }
      for (let i = 0; i < this.faceNormals.length; i++) {
        const v = this.faceNormals[i];
        quat.vmult(v, v);
      }
    }
    if (offset) {
      for (let i = 0; i < n; i++) {
        const v = verts[i];
        v.vadd(offset, v);
      }
    }
  }
  pointIsInside(p) {
    const verts = this.vertices;
    const faces = this.faces;
    const normals = this.faceNormals;
    const pointInside = new Vec3();
    this.getAveragePointLocal(pointInside);
    for (let i = 0; i < this.faces.length; i++) {
      let n = normals[i];
      const v = verts[faces[i][0]];
      const vToP = new Vec3();
      p.vsub(v, vToP);
      const r1 = n.dot(vToP);
      const vToPointInside = new Vec3();
      pointInside.vsub(v, vToPointInside);
      const r2 = n.dot(vToPointInside);
      if (r1 < 0 && r2 > 0 || r1 > 0 && r2 < 0) {
        return false;
      }
    }
    return -1;
  }
  static project(shape, axis, pos, quat, result) {
    const n = shape.vertices.length;
    const localAxis = project_localAxis;
    let max = 0;
    let min = 0;
    const localOrigin = project_localOrigin;
    const vs = shape.vertices;
    localOrigin.setZero();
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
      const temp = min;
      min = max;
      max = temp;
    }
    result[0] = max;
    result[1] = min;
  }
}
const maxminA = [];
const maxminB = [];
new Vec3();
const project_localAxis = new Vec3();
const project_localOrigin = new Vec3();
class Box extends Shape {
  constructor(halfExtents) {
    super({
      type: Shape.types.BOX
    });
    this.halfExtents = halfExtents;
    this.convexPolyhedronRepresentation = null;
    this.updateConvexPolyhedronRepresentation();
    this.updateBoundingSphereRadius();
  }
  updateConvexPolyhedronRepresentation() {
    const sx = this.halfExtents.x;
    const sy = this.halfExtents.y;
    const sz = this.halfExtents.z;
    const V = Vec3;
    const vertices = [new V(-sx, -sy, -sz), new V(sx, -sy, -sz), new V(sx, sy, -sz), new V(-sx, sy, -sz), new V(-sx, -sy, sz), new V(sx, -sy, sz), new V(sx, sy, sz), new V(-sx, sy, sz)];
    const faces = [
      [3, 2, 1, 0],
      [4, 5, 6, 7],
      [5, 4, 0, 1],
      [2, 3, 7, 6],
      [0, 4, 7, 3],
      [1, 2, 6, 5]
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
  calculateLocalInertia(mass, target) {
    if (target === void 0) {
      target = new Vec3();
    }
    Box.calculateInertia(this.halfExtents, mass, target);
    return target;
  }
  static calculateInertia(halfExtents, mass, target) {
    const e = halfExtents;
    target.x = 1 / 12 * mass * (2 * e.y * 2 * e.y + 2 * e.z * 2 * e.z);
    target.y = 1 / 12 * mass * (2 * e.x * 2 * e.x + 2 * e.z * 2 * e.z);
    target.z = 1 / 12 * mass * (2 * e.y * 2 * e.y + 2 * e.x * 2 * e.x);
  }
  getSideNormals(sixTargetVectors, quat) {
    const sides = sixTargetVectors;
    const ex = this.halfExtents;
    sides[0].set(ex.x, 0, 0);
    sides[1].set(0, ex.y, 0);
    sides[2].set(0, 0, ex.z);
    sides[3].set(-ex.x, 0, 0);
    sides[4].set(0, -ex.y, 0);
    sides[5].set(0, 0, -ex.z);
    if (quat !== void 0) {
      for (let i = 0; i !== sides.length; i++) {
        quat.vmult(sides[i], sides[i]);
      }
    }
    return sides;
  }
  volume() {
    return 8 * this.halfExtents.x * this.halfExtents.y * this.halfExtents.z;
  }
  updateBoundingSphereRadius() {
    this.boundingSphereRadius = this.halfExtents.length();
  }
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
      const wc2 = worldCornersTemp[i];
      quat.vmult(wc2, wc2);
      pos.vadd(wc2, wc2);
      const x = wc2.x;
      const y = wc2.y;
      const z = wc2.z;
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
    }
  }
}
const worldCornerTempPos = new Vec3();
const worldCornersTemp = [new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3(), new Vec3()];
const BODY_TYPES = {
  DYNAMIC: 1,
  STATIC: 2,
  KINEMATIC: 4
};
const BODY_SLEEP_STATES = {
  AWAKE: 0,
  SLEEPY: 1,
  SLEEPING: 2
};
class Body extends EventTarget {
  constructor(options) {
    if (options === void 0) {
      options = {};
    }
    super();
    this.id = Body.idCounter++;
    this.index = -1;
    this.world = null;
    this.vlambda = new Vec3();
    this.collisionFilterGroup = typeof options.collisionFilterGroup === "number" ? options.collisionFilterGroup : 1;
    this.collisionFilterMask = typeof options.collisionFilterMask === "number" ? options.collisionFilterMask : -1;
    this.collisionResponse = typeof options.collisionResponse === "boolean" ? options.collisionResponse : true;
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
    const mass = typeof options.mass === "number" ? options.mass : 0;
    this.mass = mass;
    this.invMass = mass > 0 ? 1 / mass : 0;
    this.material = options.material || null;
    this.linearDamping = typeof options.linearDamping === "number" ? options.linearDamping : 0.01;
    this.type = mass <= 0 ? Body.STATIC : Body.DYNAMIC;
    if (typeof options.type === typeof Body.STATIC) {
      this.type = options.type;
    }
    this.allowSleep = typeof options.allowSleep !== "undefined" ? options.allowSleep : true;
    this.sleepState = Body.AWAKE;
    this.sleepSpeedLimit = typeof options.sleepSpeedLimit !== "undefined" ? options.sleepSpeedLimit : 0.1;
    this.sleepTimeLimit = typeof options.sleepTimeLimit !== "undefined" ? options.sleepTimeLimit : 1;
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
    this.fixedRotation = typeof options.fixedRotation !== "undefined" ? options.fixedRotation : false;
    this.angularDamping = typeof options.angularDamping !== "undefined" ? options.angularDamping : 0.01;
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
  wakeUp() {
    const prevState = this.sleepState;
    this.sleepState = Body.AWAKE;
    this.wakeUpAfterNarrowphase = false;
    if (prevState === Body.SLEEPING) {
      this.dispatchEvent(Body.wakeupEvent);
    }
  }
  sleep() {
    this.sleepState = Body.SLEEPING;
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);
    this.wakeUpAfterNarrowphase = false;
  }
  sleepTick(time) {
    if (this.allowSleep) {
      const sleepState = this.sleepState;
      const speedSquared = this.velocity.lengthSquared() + this.angularVelocity.lengthSquared();
      const speedLimitSquared = this.sleepSpeedLimit ** 2;
      if (sleepState === Body.AWAKE && speedSquared < speedLimitSquared) {
        this.sleepState = Body.SLEEPY;
        this.timeLastSleepy = time;
        this.dispatchEvent(Body.sleepyEvent);
      } else if (sleepState === Body.SLEEPY && speedSquared > speedLimitSquared) {
        this.wakeUp();
      } else if (sleepState === Body.SLEEPY && time - this.timeLastSleepy > this.sleepTimeLimit) {
        this.sleep();
        this.dispatchEvent(Body.sleepEvent);
      }
    }
  }
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
  pointToLocalFrame(worldPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    worldPoint.vsub(this.position, result);
    this.quaternion.conjugate().vmult(result, result);
    return result;
  }
  vectorToLocalFrame(worldVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    this.quaternion.conjugate().vmult(worldVector, result);
    return result;
  }
  pointToWorldFrame(localPoint, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    this.quaternion.vmult(localPoint, result);
    result.vadd(this.position, result);
    return result;
  }
  vectorToWorldFrame(localVector, result) {
    if (result === void 0) {
      result = new Vec3();
    }
    this.quaternion.vmult(localVector, result);
    return result;
  }
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
  removeShape(shape) {
    const index = this.shapes.indexOf(shape);
    if (index === -1) {
      console.warn("Shape does not belong to the body");
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
      const shape = shapes[i];
      bodyQuat.vmult(shapeOffsets[i], offset);
      offset.vadd(this.position, offset);
      bodyQuat.mult(shapeOrientations[i], orientation);
      shape.calculateWorldAABB(offset, orientation, shapeAABB.lowerBound, shapeAABB.upperBound);
      if (i === 0) {
        aabb.copy(shapeAABB);
      } else {
        aabb.extend(shapeAABB);
      }
    }
    this.aabbNeedsUpdate = false;
  }
  updateInertiaWorld(force) {
    const I = this.invInertia;
    if (I.x === I.y && I.y === I.z && !force)
      ;
    else {
      const m1 = uiw_m1;
      const m2 = uiw_m2;
      uiw_m3;
      m1.setRotationFromQuaternion(this.quaternion);
      m1.transpose(m2);
      m1.scale(I, m1);
      m1.mmult(m2, this.invInertiaWorld);
    }
  }
  applyForce(force, relativePoint) {
    if (relativePoint === void 0) {
      relativePoint = new Vec3();
    }
    if (this.type !== Body.DYNAMIC) {
      return;
    }
    if (this.sleepState === Body.SLEEPING) {
      this.wakeUp();
    }
    const rotForce = Body_applyForce_rotForce;
    relativePoint.cross(force, rotForce);
    this.force.vadd(force, this.force);
    this.torque.vadd(rotForce, this.torque);
  }
  applyLocalForce(localForce, localPoint) {
    if (localPoint === void 0) {
      localPoint = new Vec3();
    }
    if (this.type !== Body.DYNAMIC) {
      return;
    }
    const worldForce = Body_applyLocalForce_worldForce;
    const relativePointWorld = Body_applyLocalForce_relativePointWorld;
    this.vectorToWorldFrame(localForce, worldForce);
    this.vectorToWorldFrame(localPoint, relativePointWorld);
    this.applyForce(worldForce, relativePointWorld);
  }
  applyTorque(torque) {
    if (this.type !== Body.DYNAMIC) {
      return;
    }
    if (this.sleepState === Body.SLEEPING) {
      this.wakeUp();
    }
    this.torque.vadd(torque, this.torque);
  }
  applyImpulse(impulse, relativePoint) {
    if (relativePoint === void 0) {
      relativePoint = new Vec3();
    }
    if (this.type !== Body.DYNAMIC) {
      return;
    }
    if (this.sleepState === Body.SLEEPING) {
      this.wakeUp();
    }
    const r = relativePoint;
    const velo = Body_applyImpulse_velo;
    velo.copy(impulse);
    velo.scale(this.invMass, velo);
    this.velocity.vadd(velo, this.velocity);
    const rotVelo = Body_applyImpulse_rotVelo;
    r.cross(impulse, rotVelo);
    this.invInertiaWorld.vmult(rotVelo, rotVelo);
    this.angularVelocity.vadd(rotVelo, this.angularVelocity);
  }
  applyLocalImpulse(localImpulse, localPoint) {
    if (localPoint === void 0) {
      localPoint = new Vec3();
    }
    if (this.type !== Body.DYNAMIC) {
      return;
    }
    const worldImpulse = Body_applyLocalImpulse_worldImpulse;
    const relativePointWorld = Body_applyLocalImpulse_relativePoint;
    this.vectorToWorldFrame(localImpulse, worldImpulse);
    this.vectorToWorldFrame(localPoint, relativePointWorld);
    this.applyImpulse(worldImpulse, relativePointWorld);
  }
  updateMassProperties() {
    const halfExtents = Body_updateMassProperties_halfExtents;
    this.invMass = this.mass > 0 ? 1 / this.mass : 0;
    const I = this.inertia;
    const fixed = this.fixedRotation;
    this.updateAABB();
    halfExtents.set((this.aabb.upperBound.x - this.aabb.lowerBound.x) / 2, (this.aabb.upperBound.y - this.aabb.lowerBound.y) / 2, (this.aabb.upperBound.z - this.aabb.lowerBound.z) / 2);
    Box.calculateInertia(halfExtents, this.mass, I);
    this.invInertia.set(I.x > 0 && !fixed ? 1 / I.x : 0, I.y > 0 && !fixed ? 1 / I.y : 0, I.z > 0 && !fixed ? 1 / I.z : 0);
    this.updateInertiaWorld(true);
  }
  getVelocityAtWorldPoint(worldPoint, result) {
    const r = new Vec3();
    worldPoint.vsub(this.position, r);
    this.angularVelocity.cross(r, result);
    this.velocity.vadd(result, result);
    return result;
  }
  integrate(dt, quatNormalize, quatNormalizeFast) {
    this.previousPosition.copy(this.position);
    this.previousQuaternion.copy(this.quaternion);
    if (!(this.type === Body.DYNAMIC || this.type === Body.KINEMATIC) || this.sleepState === Body.SLEEPING) {
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
    angularVelo.z += dt * (e[6] * tx + e[7] * ty + e[8] * tz);
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
    this.aabbNeedsUpdate = true;
    this.updateInertiaWorld();
  }
}
Body.idCounter = 0;
Body.COLLIDE_EVENT_NAME = "collide";
Body.DYNAMIC = BODY_TYPES.DYNAMIC;
Body.STATIC = BODY_TYPES.STATIC;
Body.KINEMATIC = BODY_TYPES.KINEMATIC;
Body.AWAKE = BODY_SLEEP_STATES.AWAKE;
Body.SLEEPY = BODY_SLEEP_STATES.SLEEPY;
Body.SLEEPING = BODY_SLEEP_STATES.SLEEPING;
Body.wakeupEvent = {
  type: "wakeup"
};
Body.sleepyEvent = {
  type: "sleepy"
};
Body.sleepEvent = {
  type: "sleep"
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
class RaycastResult {
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
  abort() {
    this.shouldStop = true;
  }
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
const RAY_MODES = {
  CLOSEST: 1,
  ANY: 2,
  ALL: 4
};
_Shape$types$SPHERE = Shape.types.SPHERE;
_Shape$types$PLANE = Shape.types.PLANE;
_Shape$types$BOX = Shape.types.BOX;
_Shape$types$CYLINDER = Shape.types.CYLINDER;
_Shape$types$CONVEXPO = Shape.types.CONVEXPOLYHEDRON;
_Shape$types$HEIGHTFI = Shape.types.HEIGHTFIELD;
_Shape$types$TRIMESH = Shape.types.TRIMESH;
class Ray {
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
    this.precision = 1e-4;
    this.checkCollisionResponse = true;
    this.skipBackfaces = false;
    this.collisionFilterMask = -1;
    this.collisionFilterGroup = -1;
    this.mode = Ray.ANY;
    this.result = new RaycastResult();
    this.hasHit = false;
    this.callback = (result) => {
    };
  }
  intersectWorld(world, options) {
    this.mode = options.mode || Ray.ANY;
    this.result = options.result || new RaycastResult();
    this.skipBackfaces = !!options.skipBackfaces;
    this.collisionFilterMask = typeof options.collisionFilterMask !== "undefined" ? options.collisionFilterMask : -1;
    this.collisionFilterGroup = typeof options.collisionFilterGroup !== "undefined" ? options.collisionFilterGroup : -1;
    this.checkCollisionResponse = typeof options.checkCollisionResponse !== "undefined" ? options.checkCollisionResponse : true;
    if (options.from) {
      this.from.copy(options.from);
    }
    if (options.to) {
      this.to.copy(options.to);
    }
    this.callback = options.callback || (() => {
    });
    this.hasHit = false;
    this.result.reset();
    this.updateDirection();
    this.getAABB(tmpAABB$1);
    tmpArray.length = 0;
    world.broadphase.aabbQuery(world, tmpAABB$1, tmpArray);
    this.intersectBodies(tmpArray);
    return this.hasHit;
  }
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
        continue;
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
  intersectBodies(bodies, result) {
    if (result) {
      this.result = result;
      this.updateDirection();
    }
    for (let i = 0, l = bodies.length; !this.result.shouldStop && i < l; i++) {
      this.intersectBody(bodies[i]);
    }
  }
  updateDirection() {
    this.to.vsub(this.from, this.direction);
    this.direction.normalize();
  }
  intersectShape(shape, quat, position, body) {
    const from = this.from;
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
    const direction = this.direction;
    const worldNormal = new Vec3(0, 0, 1);
    quat.vmult(worldNormal, worldNormal);
    const len = new Vec3();
    from.vsub(position, len);
    const planeToFrom = len.dot(worldNormal);
    to.vsub(position, len);
    const planeToTo = len.dot(worldNormal);
    if (planeToFrom * planeToTo > 0) {
      return;
    }
    if (from.distanceTo(to) < planeToFrom) {
      return;
    }
    const n_dot_dir = worldNormal.dot(direction);
    if (Math.abs(n_dot_dir) < this.precision) {
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
    shape.elementSize;
    const localRay = intersectHeightfield_localRay;
    localRay.from.copy(this.from);
    localRay.to.copy(this.to);
    Transform.pointToLocalFrame(position, quat, localRay.from, localRay.from);
    Transform.pointToLocalFrame(position, quat, localRay.to, localRay.to);
    localRay.updateDirection();
    const index = intersectHeightfield_index;
    let iMinX;
    let iMinY;
    let iMaxX;
    let iMaxY;
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
        }
        shape.getConvexTrianglePillar(i, j, false);
        Transform.pointToWorldFrame(position, quat, shape.pillarOffset, worldPillarOffset);
        this._intersectConvex(shape.pillarConvex, quat, worldPillarOffset, body, reportedShape, intersectConvexOptions);
        if (this.result.shouldStop) {
          return;
        }
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
    const a2 = (to.x - from.x) ** 2 + (to.y - from.y) ** 2 + (to.z - from.z) ** 2;
    const b2 = 2 * ((to.x - from.x) * (from.x - position.x) + (to.y - from.y) * (from.y - position.y) + (to.z - from.z) * (from.z - position.z));
    const c2 = (from.x - position.x) ** 2 + (from.y - position.y) ** 2 + (from.z - position.z) ** 2 - r ** 2;
    const delta = b2 ** 2 - 4 * a2 * c2;
    const intersectionPoint = Ray_intersectSphere_intersectionPoint;
    const normal = Ray_intersectSphere_normal;
    if (delta < 0) {
      return;
    } else if (delta === 0) {
      from.lerp(to, delta, intersectionPoint);
      intersectionPoint.vsub(position, normal);
      normal.normalize();
      this.reportIntersection(normal, intersectionPoint, reportedShape, body, -1);
    } else {
      const d1 = (-b2 - Math.sqrt(delta)) / (2 * a2);
      const d2 = (-b2 + Math.sqrt(delta)) / (2 * a2);
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
    const faceList = options && options.faceList || null;
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
      const x = position;
      vector.copy(vertices[face[0]]);
      q.vmult(vector, vector);
      vector.vadd(x, vector);
      vector.vsub(from, vector);
      q.vmult(faceNormal, normal);
      const dot = direction.dot(normal);
      if (Math.abs(dot) < this.precision) {
        continue;
      }
      const scalar = normal.dot(vector) / dot;
      if (scalar < 0) {
        continue;
      }
      direction.scale(scalar, intersectPoint);
      intersectPoint.vadd(from, intersectPoint);
      a.copy(vertices[face[0]]);
      q.vmult(a, a);
      x.vadd(a, a);
      for (let i = 1; !result.shouldStop && i < face.length - 1; i++) {
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
      }
    }
  }
  _intersectTrimesh(mesh, quat, position, body, reportedShape, options) {
    const normal = intersectTrimesh_normal;
    const triangles = intersectTrimesh_triangles;
    const treeTransform = intersectTrimesh_treeTransform;
    const vector = intersectConvex_vector;
    const localDirection = intersectTrimesh_localDirection;
    const localFrom = intersectTrimesh_localFrom;
    const localTo = intersectTrimesh_localTo;
    const worldIntersectPoint = intersectTrimesh_worldIntersectPoint;
    const worldNormal = intersectTrimesh_worldNormal;
    const indices = mesh.indices;
    mesh.vertices;
    const from = this.from;
    const to = this.to;
    const direction = this.direction;
    treeTransform.position.copy(position);
    treeTransform.quaternion.copy(quat);
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
      mesh.getNormal(trianglesIndex, normal);
      mesh.getVertex(indices[trianglesIndex * 3], a);
      a.vsub(localFrom, vector);
      const dot = localDirection.dot(normal);
      const scalar = normal.dot(vector) / dot;
      if (scalar < 0) {
        continue;
      }
      localDirection.scale(scalar, intersectPoint);
      intersectPoint.vadd(localFrom, intersectPoint);
      mesh.getVertex(indices[trianglesIndex * 3 + 1], b);
      mesh.getVertex(indices[trianglesIndex * 3 + 2], c);
      const squaredDistance = intersectPoint.distanceSquared(localFrom);
      if (!(Ray.pointInTriangle(intersectPoint, b, a, c) || Ray.pointInTriangle(intersectPoint, a, b, c)) || squaredDistance > fromToDistanceSquared) {
        continue;
      }
      Transform.vectorToWorldFrame(quat, normal, worldNormal);
      Transform.pointToWorldFrame(position, quat, intersectPoint, worldIntersectPoint);
      this.reportIntersection(worldNormal, worldIntersectPoint, reportedShape, body, trianglesIndex);
    }
    triangles.length = 0;
  }
  reportIntersection(normal, hitPointWorld, shape, body, hitFaceIndex) {
    const from = this.from;
    const to = this.to;
    const distance = from.distanceTo(hitPointWorld);
    const result = this.result;
    if (this.skipBackfaces && normal.dot(this.direction) > 0) {
      return;
    }
    result.hitFaceIndex = typeof hitFaceIndex !== "undefined" ? hitFaceIndex : -1;
    switch (this.mode) {
      case Ray.ALL:
        this.hasHit = true;
        result.set(from, to, normal, hitPointWorld, shape, body, distance);
        result.hasHit = true;
        this.callback(result);
        break;
      case Ray.CLOSEST:
        if (distance < result.distance || !result.hasHit) {
          this.hasHit = true;
          result.hasHit = true;
          result.set(from, to, normal, hitPointWorld, shape, body, distance);
        }
        break;
      case Ray.ANY:
        this.hasHit = true;
        result.hasHit = true;
        result.set(from, to, normal, hitPointWorld, shape, body, distance);
        result.shouldStop = true;
        break;
    }
  }
  static pointInTriangle(p, a2, b2, c2) {
    c2.vsub(a2, v0);
    b2.vsub(a2, v1);
    p.vsub(a2, v2);
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
  position.vsub(from, v0);
  const dot = v0.dot(direction);
  direction.scale(dot, intersect);
  intersect.vadd(from, intersect);
  const distance = position.distanceTo(intersect);
  return distance;
}
class Utils {
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
class Constraint {
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
  update() {
    throw new Error("method update() not implmemented in this Constraint subclass!");
  }
  enable() {
    const eqs = this.equations;
    for (let i = 0; i < eqs.length; i++) {
      eqs[i].enabled = true;
    }
  }
  disable() {
    const eqs = this.equations;
    for (let i = 0; i < eqs.length; i++) {
      eqs[i].enabled = false;
    }
  }
}
Constraint.idCounter = 0;
class JacobianElement {
  constructor() {
    this.spatial = new Vec3();
    this.rotational = new Vec3();
  }
  multiplyElement(element) {
    return element.spatial.dot(this.spatial) + element.rotational.dot(this.rotational);
  }
  multiplyVectors(spatial, rotational) {
    return spatial.dot(this.spatial) + rotational.dot(this.rotational);
  }
}
class Equation {
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
    this.a = 0;
    this.b = 0;
    this.eps = 0;
    this.jacobianElementA = new JacobianElement();
    this.jacobianElementB = new JacobianElement();
    this.enabled = true;
    this.multiplier = 0;
    this.setSpookParams(1e7, 4, 1 / 60);
  }
  setSpookParams(stiffness, relaxation, timeStep) {
    const d = relaxation;
    const k = stiffness;
    const h = timeStep;
    this.a = 4 / (h * (1 + 4 * d));
    this.b = 4 * d / (1 + 4 * d);
    this.eps = 4 / (h * h * k * (1 + 4 * d));
  }
  computeB(a2, b2, h) {
    const GW = this.computeGW();
    const Gq = this.computeGq();
    const GiMf = this.computeGiMf();
    return -Gq * a2 - GW * b2 - GiMf * h;
  }
  computeGq() {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const xi = bi.position;
    const xj = bj.position;
    return GA.spatial.dot(xi) + GB.spatial.dot(xj);
  }
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
  addToWlambda(deltalambda) {
    const GA = this.jacobianElementA;
    const GB = this.jacobianElementB;
    const bi = this.bi;
    const bj = this.bj;
    const temp = addToWlambda_temp;
    bi.vlambda.addScaledVector(bi.invMassSolve * deltalambda, GA.spatial, bi.vlambda);
    bj.vlambda.addScaledVector(bj.invMassSolve * deltalambda, GB.spatial, bj.vlambda);
    bi.invInertiaWorldSolve.vmult(GA.rotational, temp);
    bi.wlambda.addScaledVector(deltalambda, temp, bi.wlambda);
    bj.invInertiaWorldSolve.vmult(GB.rotational, temp);
    bj.wlambda.addScaledVector(deltalambda, temp, bj.wlambda);
  }
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
class ContactEquation extends Equation {
  constructor(bodyA, bodyB, maxForce) {
    if (maxForce === void 0) {
      maxForce = 1e6;
    }
    super(bodyA, bodyB, 0, maxForce);
    this.restitution = 0;
    this.ri = new Vec3();
    this.rj = new Vec3();
    this.ni = new Vec3();
  }
  computeB(h) {
    const a2 = this.a;
    const b2 = this.b;
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
    const n = this.ni;
    ri.cross(n, rixn);
    rj.cross(n, rjxn);
    n.negate(GA.spatial);
    rixn.negate(GA.rotational);
    GB.spatial.copy(n);
    GB.rotational.copy(rjxn);
    penetrationVec.copy(bj.position);
    penetrationVec.vadd(rj, penetrationVec);
    penetrationVec.vsub(bi.position, penetrationVec);
    penetrationVec.vsub(ri, penetrationVec);
    const g = n.dot(penetrationVec);
    const ePlusOne = this.restitution + 1;
    const GW = ePlusOne * vj.dot(n) - ePlusOne * vi.dot(n) + wj.dot(rjxn) - wi.dot(rixn);
    const GiMf = this.computeGiMf();
    const B = -g * a2 - GW * b2 - h * GiMf;
    return B;
  }
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
const ContactEquation_computeB_temp1 = new Vec3();
const ContactEquation_computeB_temp2 = new Vec3();
const ContactEquation_computeB_temp3 = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_vi = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_vj = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_xi = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_xj = new Vec3();
const ContactEquation_getImpactVelocityAlongNormal_relVel = new Vec3();
class PointToPointConstraint extends Constraint {
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
    const z = this.equationZ = new ContactEquation(bodyA, bodyB);
    this.equations.push(x, y, z);
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
    const z = this.equationZ;
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
new Vec3();
class Sphere extends Shape {
  constructor(radius) {
    super({
      type: Shape.types.SPHERE
    });
    this.radius = radius !== void 0 ? radius : 1;
    if (this.radius < 0) {
      throw new Error("The sphere radius cannot be negative.");
    }
    this.updateBoundingSphereRadius();
  }
  calculateLocalInertia(mass, target) {
    if (target === void 0) {
      target = new Vec3();
    }
    const I = 2 * mass * this.radius * this.radius / 5;
    target.x = I;
    target.y = I;
    target.z = I;
    return target;
  }
  volume() {
    return 4 * Math.PI * Math.pow(this.radius, 3) / 3;
  }
  updateBoundingSphereRadius() {
    this.boundingSphereRadius = this.radius;
  }
  calculateWorldAABB(pos, quat, min, max) {
    const r = this.radius;
    const axes = ["x", "y", "z"];
    for (let i = 0; i < axes.length; i++) {
      const ax = axes[i];
      min[ax] = pos[ax] - r;
      max[ax] = pos[ax] + r;
    }
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
class Cylinder extends ConvexPolyhedron {
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
      throw new Error("The cylinder radiusTop cannot be negative.");
    }
    if (radiusBottom < 0) {
      throw new Error("The cylinder radiusBottom cannot be negative.");
    }
    const N = numSegments;
    const vertices = [];
    const axes = [];
    const faces = [];
    const bottomface = [];
    const topface = [];
    const cos = Math.cos;
    const sin = Math.sin;
    vertices.push(new Vec3(-radiusBottom * sin(0), -height * 0.5, radiusBottom * cos(0)));
    bottomface.push(0);
    vertices.push(new Vec3(-radiusTop * sin(0), height * 0.5, radiusTop * cos(0)));
    topface.push(1);
    for (let i = 0; i < N; i++) {
      const theta = 2 * Math.PI / N * (i + 1);
      const thetaN = 2 * Math.PI / N * (i + 0.5);
      if (i < N - 1) {
        vertices.push(new Vec3(-radiusBottom * sin(theta), -height * 0.5, radiusBottom * cos(theta)));
        bottomface.push(2 * i + 2);
        vertices.push(new Vec3(-radiusTop * sin(theta), height * 0.5, radiusTop * cos(theta)));
        topface.push(2 * i + 3);
        faces.push([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2]);
      } else {
        faces.push([2 * i, 2 * i + 1, 1, 0]);
      }
      if (N % 2 === 1 || i < N / 2) {
        axes.push(new Vec3(-sin(thetaN), 0, cos(thetaN)));
      }
    }
    faces.push(bottomface);
    axes.push(new Vec3(0, 1, 0));
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
new Vec3();
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
new Vec3();
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
new Vec3();
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
new Ray();
const performance = globalThis.performance || {};
if (!performance.now) {
  let nowOffset = Date.now();
  if (performance.timing && performance.timing.navigationStart) {
    nowOffset = performance.timing.navigationStart;
  }
  performance.now = () => Date.now() - nowOffset;
}
new Vec3();
({
  type: Body.COLLIDE_EVENT_NAME,
  body: null,
  contact: null
});
function WorkerWrapper() {
  return new Worker(new URL("./assets/physicsworker.b3f26aac.js", import.meta.url), {
    "type": "module"
  });
}
const PhysicsData = Body;
const ConstraintData = PointToPointConstraint;
let log = console.log;
let report = console.error;
class Physics {
  constructor() {
    __privateAdd(this, _worker, void 0);
    __privateAdd(this, _idCounter, 0);
    __privateAdd(this, _raycastIdCounter, 0);
    __privateAdd(this, _tbuffer, new SharedArrayBuffer(4 * 16 * 1024));
    __privateAdd(this, _tview, new Float32Array(__privateGet(this, _tbuffer)));
    __privateAdd(this, _idToBody, /* @__PURE__ */ new Map());
    __privateAdd(this, _bodyToId, /* @__PURE__ */ new WeakMap());
    __privateAdd(this, _idToEntity, /* @__PURE__ */ new Map());
    __privateAdd(this, _collisionCallbacks, /* @__PURE__ */ new Map());
    __privateAdd(this, _raycastCallbacks, /* @__PURE__ */ new Map());
    __privateSet(this, _worker, new WorkerWrapper());
  }
  async init(events, logService) {
    if (logService)
      [log, report] = logService;
    log("Starting");
    events.on(`set${PhysicsData.name}Component`, (entityId, body) => {
      const bodyId = __privateGet(this, _bodyToId).get(body);
      __privateGet(this, _idToEntity).set(bodyId, entityId);
    });
    events.on(`delete${PhysicsData.name}Component`, (_, body) => {
      this.removeBody(body);
    });
    return new Promise((resolve) => {
      __privateGet(this, _worker).onerror = report;
      __privateGet(this, _worker).onmessage = ({ data }) => {
        var _a;
        if (data === "frogbones")
          log("ayye");
        switch (data.type) {
          case "ready": {
            log("\u2515 Ready");
            __privateGet(this, _worker).postMessage({ type: "init", buffer: __privateGet(this, _tbuffer) });
            resolve();
            break;
          }
          case "collisions": {
            const { collisions } = data;
            for (let i = 0; i < collisions.length; i += 2) {
              const rbId0 = collisions[i + 0];
              const rbId1 = collisions[i + 1];
              const id0 = __privateGet(this, _idToEntity).get(rbId0);
              const id1 = __privateGet(this, _idToEntity).get(rbId1);
              events.emit("collision", { id0, id1 });
              (_a = __privateGet(this, _collisionCallbacks).get(rbId0)) == null ? void 0 : _a(id1);
            }
            break;
          }
          case "raycastResult": {
            const { raycastId, bodyId, hitPoint } = data;
            const entityID = __privateGet(this, _idToEntity).get(bodyId);
            const { x, y, z } = hitPoint;
            __privateGet(this, _raycastCallbacks).get(raycastId)({
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
    for (const [id, body] of __privateGet(this, _idToBody)) {
      const offset = 3 * id;
      body.position.x = __privateGet(this, _tview)[offset + 0];
      body.position.y = __privateGet(this, _tview)[offset + 1];
      body.position.z = __privateGet(this, _tview)[offset + 2];
    }
  }
  registerCollisionCallback(body, cb) {
    const id = __privateGet(this, _bodyToId).get(body);
    __privateGet(this, _collisionCallbacks).set(id, cb);
  }
  removeCollisionCallback(body) {
    const id = __privateGet(this, _bodyToId).get(body);
    __privateGet(this, _collisionCallbacks).delete(id);
  }
  addForce(body, force) {
    __privateGet(this, _worker).postMessage({
      type: "addForce",
      id: __privateGet(this, _bodyToId).get(body),
      x: force.x,
      y: force.y,
      z: force.z
    });
  }
  addForceConditionalRaycast(body, force, from, to) {
    __privateGet(this, _worker).postMessage({
      type: "addForceConditionalRaycast",
      id: __privateGet(this, _bodyToId).get(body),
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
    __privateGet(this, _worker).postMessage({
      type: "addVelocity",
      id: __privateGet(this, _bodyToId).get(body),
      x: velocity.x,
      y: velocity.y,
      z: velocity.z
    });
  }
  addVelocityConditionalRaycast(body, velocity, from, to) {
    __privateGet(this, _worker).postMessage({
      type: "addVelocityConditionalRaycast",
      id: __privateGet(this, _bodyToId).get(body),
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
      const id = __privateGet(this, _raycastIdCounter);
      __privateSet(this, _raycastIdCounter, __privateGet(this, _raycastIdCounter) + 1);
      __privateGet(this, _raycastCallbacks).set(id, resolve);
      __privateGet(this, _worker).postMessage({
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
    const id = __privateGet(this, _bodyToId).get(body);
    __privateGet(this, _worker).postMessage({
      type: "removeBody",
      id
    });
    __privateGet(this, _bodyToId).delete(body);
    __privateGet(this, _idToBody).delete(id);
  }
  createTrimesh(opts, geometry) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m, _n, _o, _p, _q, _r, _s, _t;
    const id = __privateGet(this, _idCounter);
    __privateSet(this, _idCounter, __privateGet(this, _idCounter) + 1);
    const nonIndexedGeo = geometry.index ? geometry.toNonIndexed() : geometry;
    const triangles = nonIndexedGeo.getAttribute("position").array;
    const triangleBuffer = triangles.buffer;
    __privateGet(this, _worker).postMessage({
      type: "createTrimesh",
      triangleBuffer,
      x: (_b = (_a = opts.pos) == null ? void 0 : _a.x) != null ? _b : 0,
      y: (_d = (_c = opts.pos) == null ? void 0 : _c.y) != null ? _d : 0,
      z: (_f = (_e = opts.pos) == null ? void 0 : _e.z) != null ? _f : 0,
      sx: (_h = (_g = opts.scale) == null ? void 0 : _g.x) != null ? _h : 1,
      sy: (_j = (_i = opts.scale) == null ? void 0 : _i.y) != null ? _j : 1,
      sz: (_l = (_k = opts.scale) == null ? void 0 : _k.z) != null ? _l : 1,
      qx: (_n = (_m = opts.quat) == null ? void 0 : _m.x) != null ? _n : 0,
      qy: (_p = (_o = opts.quat) == null ? void 0 : _o.y) != null ? _p : 0,
      qz: (_r = (_q = opts.quat) == null ? void 0 : _q.z) != null ? _r : 0,
      qw: (_t = (_s = opts.quat) == null ? void 0 : _s.w) != null ? _t : 1,
      id
    }, []);
    const mock = new Body();
    __privateGet(this, _idToBody).set(id, mock);
    __privateGet(this, _bodyToId).set(mock, id);
    return mock;
  }
  createSphere(opts, radius) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m, _n, _o, _p, _q, _r, _s, _t, _u;
    const id = __privateGet(this, _idCounter);
    __privateSet(this, _idCounter, __privateGet(this, _idCounter) + 1);
    __privateGet(this, _worker).postMessage({
      type: "createSphere",
      radius,
      mass: opts.mass,
      x: (_b = (_a = opts.pos) == null ? void 0 : _a.x) != null ? _b : 0,
      y: (_d = (_c = opts.pos) == null ? void 0 : _c.y) != null ? _d : 0,
      z: (_f = (_e = opts.pos) == null ? void 0 : _e.z) != null ? _f : 0,
      sx: (_h = (_g = opts.scale) == null ? void 0 : _g.x) != null ? _h : 1,
      sy: (_j = (_i = opts.scale) == null ? void 0 : _i.y) != null ? _j : 1,
      sz: (_l = (_k = opts.scale) == null ? void 0 : _k.z) != null ? _l : 1,
      qx: (_n = (_m = opts.quat) == null ? void 0 : _m.x) != null ? _n : 0,
      qy: (_p = (_o = opts.quat) == null ? void 0 : _o.y) != null ? _p : 0,
      qz: (_r = (_q = opts.quat) == null ? void 0 : _q.z) != null ? _r : 0,
      qw: (_t = (_s = opts.quat) == null ? void 0 : _s.w) != null ? _t : 1,
      fixedRotation: (_u = opts.fixedRotation) != null ? _u : false,
      id
    });
    const mock = new Body();
    __privateGet(this, _idToBody).set(id, mock);
    __privateGet(this, _bodyToId).set(mock, id);
    return mock;
  }
  createCapsule(opts, radius, height) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m, _n, _o, _p, _q, _r, _s, _t, _u;
    const id = __privateGet(this, _idCounter);
    __privateSet(this, _idCounter, __privateGet(this, _idCounter) + 1);
    __privateGet(this, _worker).postMessage({
      type: "createCapsule",
      radius,
      height,
      mass: opts.mass,
      x: (_b = (_a = opts.pos) == null ? void 0 : _a.x) != null ? _b : 0,
      y: (_d = (_c = opts.pos) == null ? void 0 : _c.y) != null ? _d : 0,
      z: (_f = (_e = opts.pos) == null ? void 0 : _e.z) != null ? _f : 0,
      sx: (_h = (_g = opts.scale) == null ? void 0 : _g.x) != null ? _h : 1,
      sy: (_j = (_i = opts.scale) == null ? void 0 : _i.y) != null ? _j : 1,
      sz: (_l = (_k = opts.scale) == null ? void 0 : _k.z) != null ? _l : 1,
      qx: (_n = (_m = opts.quat) == null ? void 0 : _m.x) != null ? _n : 0,
      qy: (_p = (_o = opts.quat) == null ? void 0 : _o.y) != null ? _p : 0,
      qz: (_r = (_q = opts.quat) == null ? void 0 : _q.z) != null ? _r : 0,
      qw: (_t = (_s = opts.quat) == null ? void 0 : _s.w) != null ? _t : 1,
      fixedRotation: (_u = opts.fixedRotation) != null ? _u : false,
      id
    });
    const mock = new Body();
    __privateGet(this, _idToBody).set(id, mock);
    __privateGet(this, _bodyToId).set(mock, id);
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
_worker = new WeakMap();
_idCounter = new WeakMap();
_raycastIdCounter = new WeakMap();
_tbuffer = new WeakMap();
_tview = new WeakMap();
_idToBody = new WeakMap();
_bodyToId = new WeakMap();
_idToEntity = new WeakMap();
_collisionCallbacks = new WeakMap();
_raycastCallbacks = new WeakMap();
export { ConstraintData, Physics, PhysicsData };
