/// <reference types="node" />
import { BufferGeometry } from 'three';
import EventEmitter from 'events';
declare type LogFn = (payload: object | string | number) => void;
export declare type Vec3 = [number, number, number];
export declare type Quat = [number, number, number, number];
export declare type RaycastResult = {
    entityID: number;
    hitPoint: Vec3;
};
export declare type RigidBodyOptions = {
    pos?: Vec3;
    scale?: Vec3;
    quat?: Quat;
    mass?: number;
    fixedRotation?: boolean;
};
export declare type CollisionCallback = (entity: number) => void;
export declare class PhysicsData {
    id: number;
    constructor(id: number);
}
export declare class Physics {
    #private;
    constructor();
    init(events: EventEmitter, logService?: LogFn[], workerLogService?: LogFn[]): Promise<void>;
    update(): void;
    getBodyPosition({ id }: PhysicsData): number[];
    registerCollisionCallback({ id }: PhysicsData, cb: CollisionCallback): void;
    removeCollisionCallback({ id }: PhysicsData): void;
    addForce({ id }: PhysicsData, force: Vec3): void;
    addForceConditionalRaycast({ id }: PhysicsData, force: Vec3, from: Vec3, to: Vec3): void;
    addVelocity({ id }: PhysicsData, velocity: Vec3): void;
    /** Adds velocity to a RigidBody ONLY if raycast returns a hit */
    addVelocityConditionalRaycast({ id }: PhysicsData, velocity: Vec3, from: Vec3, to: Vec3): void;
    /** Casts a ray, and returns either the entity ID that got hit or undefined. */
    raycast(from: Vec3, to: Vec3): Promise<RaycastResult | null>;
    removeBody({ id }: PhysicsData): void;
    createTrimesh(opts: RigidBodyOptions, geometry: BufferGeometry): PhysicsData;
    createPlane(opts: RigidBodyOptions): {
        id: number;
    };
    createSphere(opts: RigidBodyOptions & {
        radius: number;
    }): PhysicsData;
    createCapsule(opts: RigidBodyOptions & {
        radius: number;
        height: number;
    }): PhysicsData;
}
export {};
