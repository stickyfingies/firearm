import { BufferGeometry } from 'three';
import Backend from './physicsworker?worker';
import EventEmitter from 'events';

/* --------------------------------- TYPES --------------------------------- */

type RigidBodyID
    = number
    ;
type RaycastCallback
    = (i: RaycastResult | null) => void
    ;
type LogFn
    = (payload: object | string | number) => void
    ;

/* -------------------------------- GLOBALS -------------------------------- */

let [log, report]
    : LogFn[]
    = [console.log, console.error]
    ;
let [workerLog, workerReport]
    : LogFn[]
    = [console.log, console.error]
    ;

/* ------------------------------ PUBLIC TYPES ----------------------------- */

export type Vec3
    = [number, number, number]
    ;
export type Quat
    = [number, number, number, number]
    ;
export type RaycastResult
    =
    {
        entityID: number, // the entity hit by the raycast
        hitPoint: Vec3 // hitpoint lcoation in worldspace
    }
    ;
export type RigidBodyOptions
    = {
        pos?: Vec3,
        scale?: Vec3,
        quat?: Quat,
        mass?: number,
        fixedRotation?: boolean
    }
    ;
export type CollisionCallback
    =
    (entity: number) => void
    ;

/* ----------------------------- PUBLIC CLASSES ---------------------------- */

export class PhysicsData { constructor(public id: number) { } }
export class Physics {
    #worker: Worker;

    /** A counter that gets incremented when a new ID needs to be allocated. */
    #idCounter: RigidBodyID = 0;

    #raycastIdCounter = 0;

    /** Raw buffer containing RigidBody transform data. */
    #tbuffer = new SharedArrayBuffer(4 * 16 * 1024);

    #tview = new Float32Array(this.#tbuffer);

    /** Map of RigidBody ID's to Entity ID's */
    #idToEntity = new Map<RigidBodyID, number>();

    #collisionCallbacks = new Map<RigidBodyID, CollisionCallback>();

    #raycastCallbacks = new Map<number, RaycastCallback>();

    constructor() {
        this.#worker = new Backend();
    }

    async init(events: EventEmitter, logService?: LogFn[], workerLogService?: LogFn[]) {
        if (logService) {
            [log, report] = logService;
            [workerLog, workerReport] = logService;
        }
        if (workerLogService) [workerLog, workerReport] = workerLogService;

        log(`${import.meta.url}`);

        events.on(`set${PhysicsData.name}Component`, (entityId: number, { id }: PhysicsData) => {
            this.#idToEntity.set(id, entityId);
        });
        events.on(`delete${PhysicsData.name}Component`, (_, body: PhysicsData) => {
            this.removeBody(body);
        });

        return new Promise<void>((resolve) => {
            this.#worker.onerror = workerReport;
            this.#worker.onmessage = ({ data }) => {
                switch (data.type) {
                    case 'log': {
                        workerLog(data.message);
                        break;
                    }
                    case 'ready': {
                        log('Ready');
                        // This is the backend saying, "libraries loaded and ready to go!"
                        this.#worker.postMessage({ type: 'init', buffer: this.#tbuffer });
                        resolve();
                        break;
                    }
                    case 'collisions': {
                        // List of collisions that take place every tick
                        const { collisions } = data;
                        for (let i = 0; i < collisions.length; i += 2) {
                            const rbId0 = collisions[i + 0];
                            const rbId1 = collisions[i + 1];

                            const id0 = this.#idToEntity.get(rbId0)!;
                            const id1 = this.#idToEntity.get(rbId1)!;

                            events.emit('collision', { id0, id1 });
                            this.#collisionCallbacks.get(rbId0)?.(id1);
                        }
                        break;
                    }
                    case 'raycastResult': {
                        // Results from a raycast request
                        const { raycastId, bodyId, hitPoint } = data;
                        const didHit = (bodyId !== -1);
                        if (didHit) {
                            const entityID = this.#idToEntity.get(bodyId)!;
                            const { x, y, z } = hitPoint;
                            this.#raycastCallbacks.get(raycastId)!({
                                entityID,
                                hitPoint: [x, y, z],
                            });
                        }
                        else {
                            this.#raycastCallbacks.get(raycastId)!(null);
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
        // for (const [id, body] of this.#idToBody) {
        //     const offset = 3 * id;
        //     body.position.x = this.#tview[offset + 0];
        //     body.position.y = this.#tview[offset + 1];
        //     body.position.z = this.#tview[offset + 2];
        // }
    }

    // I Want to move away from cannon-es, so I'm trying to remove all its types.
    // That means walking out on Vec3 and similar, so I'm using arrays to encode
    // transform data.

    // That sucks on the engine end, because functions like `distanceTo` and `vsub`
    // aren't available anymore without wrapping the array... gross.
    getBodyPosition({ id }: PhysicsData): Vec3 {
        const offset = 3 * id;
        return Array.from(this.#tview.slice(offset, offset + 3)) as Vec3;
    }

    registerCollisionCallback({ id }: PhysicsData, cb: CollisionCallback) {
        this.#collisionCallbacks.set(id, cb);
    }

    removeCollisionCallback({ id }: PhysicsData) {
        this.#collisionCallbacks.delete(id);
    }

    addForce({ id }: PhysicsData, force: Vec3) {
        this.#worker.postMessage({
            type: 'addForce',
            id,
            x: force[0],
            y: force[1],
            z: force[2],
        });
    }

    addForceConditionalRaycast({ id }: PhysicsData, force: Vec3, from: Vec3, to: Vec3) {
        this.#worker.postMessage({
            type: 'addForceConditionalRaycast',
            id,
            x: force[0],
            y: force[1],
            z: force[2],
            fx: from[0],
            fy: from[1],
            fz: from[2],
            tx: to[0],
            ty: to[1],
            tz: to[2],
        });
    }

    addVelocity({ id }: PhysicsData, velocity: Vec3) {
        this.#worker.postMessage({
            type: 'addVelocity',
            id,
            x: velocity[0],
            y: velocity[1],
            z: velocity[2],
        });
    }

    /** Adds velocity to a RigidBody ONLY if raycast returns a hit */
    addVelocityConditionalRaycast({ id }: PhysicsData, velocity: Vec3, from: Vec3, to: Vec3) {
        this.#worker.postMessage({
            type: 'addVelocityConditionalRaycast',
            id,
            vx: velocity[0],
            vy: velocity[1],
            vz: velocity[2],
            fx: from[0],
            fy: from[1],
            fz: from[2],
            tx: to[0],
            ty: to[1],
            tz: to[2],
        });
    }

    /** Casts a ray, and returns either the entity ID that got hit or undefined. */
    raycast(from: Vec3, to: Vec3) {
        return new Promise<RaycastResult | null>((resolve) => {
            const id = this.#raycastIdCounter;
            this.#raycastIdCounter += 1;

            this.#raycastCallbacks.set(id, resolve);

            this.#worker.postMessage({
                type: 'raycast',
                id,
                fx: from[0],
                fy: from[1],
                fz: from[2],
                tx: to[0],
                ty: to[1],
                tz: to[2],
            });
        });
    }

    removeBody({ id }: PhysicsData) {
        this.#worker.postMessage({
            type: 'removeBody',
            id,
        });
    }

    createTrimesh(opts: RigidBodyOptions, geometry: BufferGeometry): PhysicsData {
        const id = this.#idCounter;
        this.#idCounter += 1;

        // optimization: extract underlying buffer from the ThreeJS BufferAttribute
        // so that it can be moved to the worker thread, instead of copied.

        const nonIndexedGeo = geometry.index ? geometry.toNonIndexed() : geometry;
        const triangles = nonIndexedGeo.getAttribute('position').array as Float32Array;
        const triangleBuffer = triangles.buffer;

        this.#worker.postMessage({
            type: 'createTrimesh',
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
            id,
        }, []);

        return { id };
    }

    createPlane(opts: RigidBodyOptions) {
        const id = this.#idCounter;
        this.#idCounter += 1;

        this.#worker.postMessage({
            type: 'createPlane',
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
            id,
        });

        return { id };
    }

    createSphere(opts: RigidBodyOptions & { radius: number }): PhysicsData {
        const id = this.#idCounter;
        this.#idCounter += 1;

        this.#worker.postMessage({
            type: 'createSphere',
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
            id,
        });

        return { id };
    }

    createCapsule(opts: RigidBodyOptions & { radius: number, height: number }): PhysicsData {
        const id = this.#idCounter;
        this.#idCounter += 1;

        this.#worker.postMessage({
            type: 'createCapsule',
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
            id,
        });

        return { id };
    }
}
