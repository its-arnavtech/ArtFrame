from __future__ import annotations

from dataclasses import dataclass

import cv2

from app.graphics.framebuffer import CpuFramebuffer, FramebufferSpec
from app.interaction.hand_controls import InteractionState
from app.types import Point2D


@dataclass
class Particle:
    position: Point2D
    velocity: Point2D
    life: float


class ParticleField:
    """Small CPU particle proof-of-concept; simulation policies remain replaceable."""

    def __init__(self, max_particles: int = 96) -> None:
        self._max_particles = max_particles
        self._particles: list[Particle] = []

    def update(
        self,
        interaction: InteractionState,
        delta_seconds: float,
        frame_size: tuple[int, int],
    ) -> None:
        del frame_size
        updated: list[Particle] = []
        for particle in self._particles:
            life = particle.life - delta_seconds
            if life > 0.0:
                updated.append(
                    Particle(
                        position=Point2D(
                            particle.position.x + particle.velocity.x * delta_seconds,
                            particle.position.y + particle.velocity.y * delta_seconds,
                        ),
                        velocity=particle.velocity,
                        life=life,
                    )
                )
        self._particles = updated

        for hand in interaction.active_hands():
            if len(self._particles) >= self._max_particles:
                break
            self._particles.append(Particle(hand.position, hand.velocity, 0.55))

    def render(self, frame_size: tuple[int, int]) -> CpuFramebuffer:
        width, height = frame_size
        target = CpuFramebuffer(FramebufferSpec(width, height))
        for particle in self._particles:
            if not 0.0 <= particle.position.x <= 1.0 or not 0.0 <= particle.position.y <= 1.0:
                continue
            center = (int(particle.position.x * width), int(particle.position.y * height))
            alpha = int(100 * min(1.0, particle.life / 0.55))
            cv2.circle(target.color, center, 2, (235, 225, 255, alpha), thickness=-1)
        return target
