try:
    from .deformable_aggregation import (
        deformable_aggregation_func,
        feature_maps_format,
        deformable_format,
    )
except ImportError:
    deformable_aggregation_func = None
    feature_maps_format = None
    deformable_format = None
