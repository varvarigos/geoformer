"""
Manifold operations for Hyperbolic, Spherical, and Euclidean geometries.
Implements exponential and logarithmic maps, geodesic distances, and parallel transport.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class Manifold(nn.Module):
    """Base class for manifold operations."""
    
    def __init__(self, curvature: float = 1.0, learnable: bool = False, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.learnable = learnable
        
        if learnable:
            # Store raw unconstrained parameter
            # Initialize to value that gives desired curvature after transformation
            self._curvature_param = nn.Parameter(torch.tensor(0.0))
        else:
            # Fixed curvature
            self.register_buffer('_curvature_value', torch.tensor(curvature))
    
    def get_curvature(self) -> torch.Tensor:
        """Get the (potentially constrained) curvature value."""
        if self.learnable:
            # Override in subclasses to apply constraints
            return self._curvature_param
        else:
            return self._curvature_value
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: maps tangent vector v at point x to the manifold."""
        raise NotImplementedError
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: maps point y to tangent space at point x."""
        raise NotImplementedError
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between points x and y on the manifold."""
        raise NotImplementedError
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project point x onto the manifold."""
        raise NotImplementedError
    
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto tangent space at point x."""
        raise NotImplementedError


class EuclideanManifold(Manifold):
    """Euclidean space (flat geometry, κ = 0)."""
    
    def __init__(self, eps: float = 1e-7):
        super().__init__(curvature=0.0, eps=eps)
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map in Euclidean space (simple addition)."""
        return x + v
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map in Euclidean space (simple subtraction)."""
        return y - x
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Euclidean distance."""
        return torch.norm(y - x, dim=-1, keepdim=True)
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """No projection needed in Euclidean space."""
        return x
    
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """No projection needed in Euclidean space."""
        return v


class HyperbolicManifold(Manifold):
    """
    Hyperbolic space using the Poincaré ball model (κ < 0).
    The Poincaré ball is {x ∈ R^d : ||x|| < 1/sqrt(-κ)}
    """
    
    def __init__(self, curvature: float = -1.0, learnable: bool = False, eps: float = 1e-7):
        if not learnable:
            assert curvature < 0, "Hyperbolic curvature must be negative"
        super().__init__(curvature=curvature, learnable=learnable, eps=eps)
        
        if learnable:
            # Random initialization for learnable curvature
            # Sample θ uniformly from log-space to get diverse negative curvatures
            # θ ∈ [-2, 2] gives κ ∈ [-exp(2), -exp(-2)] ≈ [-7.4, -0.14]
            initial_param = torch.empty(1).uniform_(-2.0, 2.0).item()
            self._curvature_param.data.fill_(initial_param)
    
    def get_curvature(self) -> torch.Tensor:
        """Get constrained negative curvature: κ = -exp(θ)"""
        if self.learnable:
            # Constrain to negative values: κ = -exp(clamp(θ))
            clamped_param = torch.clamp(self._curvature_param, -10, 10)
            return -torch.exp(clamped_param)
        else:
            return self._curvature_value
    
    @property
    def max_norm(self):
        """Compute max norm based on current curvature."""
        curv = self.get_curvature()
        return (1. / torch.sqrt(-curv)) - self.eps
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project point onto Poincaré ball."""
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        max_norm = self.max_norm
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)
    
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space (for Poincaré ball, tangent space is R^d)."""
        return v
    
    def lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor at point x."""
        # λ(x) = 2 / (1 - κ||x||²)
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        curv = self.get_curvature()
        return 2.0 / (1.0 - curv * x_sqnorm).clamp_min(self.eps)
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map in Poincaré ball model.
        exp_x(v) = x ⊕ tanh(λ_x ||v|| / 2) * v/||v||
        where ⊕ is Möbius addition
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        lambda_x = self.lambda_x(x)
        curv = self.get_curvature()
        
        # Scale factor with clamping for numerical stability
        scale_arg = (torch.sqrt(-curv) * lambda_x * v_norm / 2.0).clamp(-15, 15)
        scale = torch.tanh(scale_arg)
        scaled_v = scale * v / v_norm
        
        # Möbius addition
        result = self.mobius_add(x, scaled_v)
        return self.proj(result)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map in Poincaré ball model.
        log_x(y) = (2 / (λ_x sqrt(-κ))) * arctanh(sqrt(-κ)||x ⊖ y||) * (x ⊖ y) / ||x ⊖ y||
        """
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp_min(self.eps)
        lambda_x = self.lambda_x(x)
        curv = self.get_curvature()
        
        # Scale factor with clamping for numerical stability
        # atanh input must be in (-1, 1)
        atanh_arg = (torch.sqrt(-curv) * diff_norm).clamp(-1.0 + self.eps, 1.0 - self.eps)
        scale = (2.0 / (lambda_x * torch.sqrt(-curv))) * torch.atanh(atanh_arg)
        
        return scale * diff / diff_norm
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition in Poincaré ball."""
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        curv = self.get_curvature()
        
        numerator = (1.0 - 2.0 * curv * xy_dot - curv * y_sqnorm) * x + \
                   (1.0 + curv * x_sqnorm) * y
        denominator = (1.0 - 2.0 * curv * xy_dot + 
                      curv * curv * x_sqnorm * y_sqnorm).clamp_min(self.eps)
        
        return numerator / denominator
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance in Poincaré ball."""
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp_min(self.eps)
        curv = self.get_curvature()
        return (2.0 / torch.sqrt(-curv)) * torch.atanh(
            torch.sqrt(-curv) * diff_norm.clamp(max=1.0 - self.eps)
        )


class SphericalManifold(Manifold):
    """
    Spherical space (κ > 0).
    The sphere is {x ∈ R^{d+1} : ||x|| = 1/sqrt(κ)}
    """
    
    def __init__(self, curvature: float = 1.0, learnable: bool = False, eps: float = 1e-7):
        if not learnable:
            assert curvature > 0, "Spherical curvature must be positive"
        super().__init__(curvature=curvature, learnable=learnable, eps=eps)
        
        if learnable:
            # Random initialization for learnable curvature
            # Sample θ uniformly from log-space to get diverse positive curvatures
            # θ ∈ [-2, 2] gives κ ∈ [exp(-2), exp(2)] ≈ [0.14, 7.4]
            initial_param = torch.empty(1).uniform_(-2.0, 2.0).item()
            self._curvature_param.data.fill_(initial_param)
    
    def get_curvature(self) -> torch.Tensor:
        """Get constrained positive curvature: κ = exp(θ)"""
        if self.learnable:
            # Constrain to positive values: κ = exp(clamp(θ))
            clamped_param = torch.clamp(self._curvature_param, -10, 10)
            return torch.exp(clamped_param)
        else:
            return self._curvature_value
    
    @property
    def radius(self):
        """Compute radius based on current curvature."""
        curv = self.get_curvature()
        return 1.0 / torch.sqrt(curv)
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project point onto sphere by normalizing."""
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        radius = self.radius
        return x / norm * radius
    
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space (orthogonal to x)."""
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        x_normalized = x / x_norm
        # v_tan = v - <v, x>x / ||x||²
        dot_product = torch.sum(v * x_normalized, dim=-1, keepdim=True)
        return v - dot_product * x_normalized
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on sphere.
        exp_x(v) = cos(||v||/R) * x + sin(||v||/R) * R * v/||v||
        where R = 1/sqrt(κ) is the radius
        """
        v = self.proj_tan(x, v)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        radius = self.radius
        
        # Angle to move along geodesic
        theta = v_norm / radius
        
        result = torch.cos(theta) * x + torch.sin(theta) * radius * v / v_norm
        return self.proj(result)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map on sphere.
        log_x(y) = R * arccos(<x,y>/R²) * (y - <x,y>x/R²) / ||y - <x,y>x/R²||
        """
        x = self.proj(x)
        y = self.proj(y)
        radius = self.radius
        
        dot_product = torch.sum(x * y, dim=-1, keepdim=True) / (radius ** 2)
        dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)
        
        # Direction in tangent space
        direction = y - dot_product * x
        direction_norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(self.eps)
        
        # Angle
        theta = torch.acos(dot_product)
        
        return radius * theta * direction / direction_norm
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance on sphere (arc length)."""
        x = self.proj(x)
        y = self.proj(y)
        radius = self.radius
        
        dot_product = torch.sum(x * y, dim=-1, keepdim=True) / (radius ** 2)
        dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)
        
        return radius * torch.acos(dot_product)


def get_manifold(curvature: float, eps: float = 1e-7) -> Manifold:
    """Factory function to create appropriate manifold based on curvature."""
    if abs(curvature) < eps:
        return EuclideanManifold(eps=eps)
    elif curvature < 0:
        return HyperbolicManifold(curvature=curvature, eps=eps)
    else:
        return SphericalManifold(curvature=curvature, eps=eps)
