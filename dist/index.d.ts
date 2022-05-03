/**
 * const lmao_bats = import.meta.url.split('?')[0].split('/');
  lmao_bats.pop();
  console.log(lmao_bats.join('/'));
 */
import { BufferGeometry } from 'three';
import { Body, PointToPointConstraint, Quaternion, Vec3 } from 'cannon-es';
import EventEmitter from 'events';
export declare const PhysicsData: typeof Body;
export declare type PhysicsData = Body;
export declare const ConstraintData: typeof PointToPointConstraint;
export declare type ConstraintData = PointToPointConstraint;
export declare type RaycastInfo = {
    entityID: number;
    hitPoint: Vec3;
};
export declare type RigidBodyOptions = {
    pos?: Vec3;
    scale?: Vec3;
    quat?: Quaternion;
    mass?: number;
    fixedRotation?: boolean;
};
export declare type CollisionCallback = (entity: number) => void;
declare type LogFn = (payload: object | string | number) => void;
export declare class Physics {
    #private;
    constructor();
    init(events: EventEmitter, logService?: LogFn[]): Promise<void>;
    update(): void;
    registerCollisionCallback(body: Body, cb: CollisionCallback): void;
    removeCollisionCallback(body: Body): void;
    addForce(body: Body, force: Vec3): void;
    addForceConditionalRaycast(body: Body, force: Vec3, from: Vec3, to: Vec3): void;
    addVelocity(body: Body, velocity: Vec3): void;
    /** Adds velocity to a RigidBody ONLY if raycast returns a hit */
    addVelocityConditionalRaycast(body: Body, velocity: Vec3, from: Vec3, to: Vec3): void;
    /** Casts a ray, and returns either the entity ID that got hit or undefined. */
    raycast(from: Vec3, to: Vec3): Promise<RaycastInfo | undefined>;
    removeBody(body: Body): void;
    createTrimesh(opts: RigidBodyOptions, geometry: BufferGeometry): Body;
    createSphere(opts: RigidBodyOptions, radius: number): Body;
    createCapsule(opts: RigidBodyOptions, radius: number, height: number): Body;
    /** @deprecated */
    static makeCube(mass: number, size: number): Body;
    /** @deprecated */
    static makeBall(mass: number, radius: number): Body;
    /** @deprecated */
    static makeCylinder(mass: number, radius: number, height: number): Body;
    /** @deprecated */
    static makeCapsule(mass: number, radius: number, height: number): Body;
}
export {};
