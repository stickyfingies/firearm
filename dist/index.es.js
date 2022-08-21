function WorkerWrapper() {
          return new Worker(new URL("./physicsworker.d99472d4.js", import.meta.url), {
  "type": "module"
})
        }

let [log, report] = [console.log, console.error];
let [workerLog, workerReport] = [console.log, console.error];
class PhysicsData {
  constructor(id) {
    this.id = id;
  }
}
class Physics {
  #worker;
  #idCounter = 0;
  #raycastIdCounter = 0;
  #tbuffer = new SharedArrayBuffer(4 * 16 * 1024);
  #tview = new Float32Array(this.#tbuffer);
  #idToEntity = /* @__PURE__ */ new Map();
  #collisionCallbacks = /* @__PURE__ */ new Map();
  #raycastCallbacks = /* @__PURE__ */ new Map();
  constructor() {
    this.#worker = new WorkerWrapper();
  }
  async init(events, logService, workerLogService) {
    if (logService) {
      [log, report] = logService;
      [workerLog, workerReport] = logService;
    }
    if (workerLogService)
      [workerLog, workerReport] = workerLogService;
    log(`${import.meta.url}`);
    events.on(`set${PhysicsData.name}Component`, (entityId, { id }) => {
      this.#idToEntity.set(id, entityId);
    });
    events.on(`delete${PhysicsData.name}Component`, (_, body) => {
      this.removeBody(body);
    });
    return new Promise((resolve) => {
      this.#worker.onerror = workerReport;
      this.#worker.onmessage = ({ data }) => {
        switch (data.type) {
          case "log": {
            workerLog(data.message);
            break;
          }
          case "ready": {
            log("Ready");
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
            const didHit = bodyId !== -1;
            if (didHit) {
              const entityID = this.#idToEntity.get(bodyId);
              const { x, y, z } = hitPoint;
              this.#raycastCallbacks.get(raycastId)({
                entityID,
                hitPoint: [x, y, z]
              });
            } else {
              this.#raycastCallbacks.get(raycastId)(null);
            }
            break;
          }
          default: {
            report(`Unknown message type ${data.type}`);
          }
        }
      };
    });
  }
  update() {
  }
  getBodyPosition({ id }) {
    const offset = 3 * id;
    return Array.from(this.#tview.slice(offset, offset + 3));
  }
  registerCollisionCallback({ id }, cb) {
    this.#collisionCallbacks.set(id, cb);
  }
  removeCollisionCallback({ id }) {
    this.#collisionCallbacks.delete(id);
  }
  addForce({ id }, force) {
    this.#worker.postMessage({
      type: "addForce",
      id,
      x: force[0],
      y: force[1],
      z: force[2]
    });
  }
  addForceConditionalRaycast({ id }, force, from, to) {
    this.#worker.postMessage({
      type: "addForceConditionalRaycast",
      id,
      x: force[0],
      y: force[1],
      z: force[2],
      fx: from[0],
      fy: from[1],
      fz: from[2],
      tx: to[0],
      ty: to[1],
      tz: to[2]
    });
  }
  addVelocity({ id }, velocity) {
    this.#worker.postMessage({
      type: "addVelocity",
      id,
      x: velocity[0],
      y: velocity[1],
      z: velocity[2]
    });
  }
  addVelocityConditionalRaycast({ id }, velocity, from, to) {
    this.#worker.postMessage({
      type: "addVelocityConditionalRaycast",
      id,
      vx: velocity[0],
      vy: velocity[1],
      vz: velocity[2],
      fx: from[0],
      fy: from[1],
      fz: from[2],
      tx: to[0],
      ty: to[1],
      tz: to[2]
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
        fx: from[0],
        fy: from[1],
        fz: from[2],
        tx: to[0],
        ty: to[1],
        tz: to[2]
      });
    });
  }
  removeBody({ id }) {
    this.#worker.postMessage({
      type: "removeBody",
      id
    });
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
      x: opts.pos?.[0] ?? 0,
      y: opts.pos?.[1] ?? 0,
      z: opts.pos?.[2] ?? 0,
      sx: opts.scale?.[0] ?? 1,
      sy: opts.scale?.[1] ?? 1,
      sz: opts.scale?.[2] ?? 1,
      qx: opts.quat?.[0] ?? 0,
      qy: opts.quat?.[1] ?? 0,
      qz: opts.quat?.[2] ?? 0,
      qw: opts.quat?.[3] ?? 1,
      id
    }, []);
    return { id };
  }
  createPlane(opts) {
    const id = this.#idCounter;
    this.#idCounter += 1;
    this.#worker.postMessage({
      type: "createPlane",
      mass: opts.mass,
      x: opts.pos?.[0] ?? 0,
      y: opts.pos?.[1] ?? 0,
      z: opts.pos?.[2] ?? 0,
      sx: opts.scale?.[0] ?? 1,
      sy: opts.scale?.[1] ?? 1,
      sz: opts.scale?.[2] ?? 1,
      qx: opts.quat?.[0] ?? 0,
      qy: opts.quat?.[1] ?? 0,
      qz: opts.quat?.[2] ?? 0,
      qw: opts.quat?.[3] ?? 1,
      fixedRotation: opts.fixedRotation ?? false,
      id
    });
    return { id };
  }
  createSphere(opts) {
    const id = this.#idCounter;
    this.#idCounter += 1;
    this.#worker.postMessage({
      type: "createSphere",
      radius: opts.radius,
      mass: opts.mass,
      x: opts.pos?.[0] ?? 0,
      y: opts.pos?.[1] ?? 0,
      z: opts.pos?.[2] ?? 0,
      sx: opts.scale?.[0] ?? 1,
      sy: opts.scale?.[1] ?? 1,
      sz: opts.scale?.[2] ?? 1,
      qx: opts.quat?.[0] ?? 0,
      qy: opts.quat?.[1] ?? 0,
      qz: opts.quat?.[2] ?? 0,
      qw: opts.quat?.[3] ?? 1,
      fixedRotation: opts.fixedRotation ?? false,
      id
    });
    return { id };
  }
  createCapsule(opts) {
    const id = this.#idCounter;
    this.#idCounter += 1;
    this.#worker.postMessage({
      type: "createCapsule",
      radius: opts.radius,
      height: opts.height,
      mass: opts.mass,
      x: opts.pos?.[0] ?? 0,
      y: opts.pos?.[1] ?? 0,
      z: opts.pos?.[2] ?? 0,
      sx: opts.scale?.[0] ?? 1,
      sy: opts.scale?.[1] ?? 1,
      sz: opts.scale?.[2] ?? 1,
      qx: opts.quat?.[0] ?? 0,
      qy: opts.quat?.[1] ?? 0,
      qz: opts.quat?.[2] ?? 0,
      qw: opts.quat?.[3] ?? 1,
      fixedRotation: opts.fixedRotation ?? false,
      id
    });
    return { id };
  }
}

export { Physics, PhysicsData };
