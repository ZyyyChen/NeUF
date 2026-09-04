from __future__ import annotations

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SliceRenderer:
    """Direct point-query renderer shared by training, evaluation, and export."""

    def __init__(self, dataset) -> None:
        if dataset is None:
            raise ValueError("SliceRenderer requires a dataset")
        self.dataset = dataset
        self.width_px = int(dataset.px_width)
        self.height_px = int(dataset.px_height)

    @staticmethod
    def _flatten(points: torch.Tensor, viewdirs: torch.Tensor):
        return (
            points.reshape(-1, points.shape[-1]),
            viewdirs.reshape(-1, viewdirs.shape[-1]),
        )

    def query_points(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        *,
        alpha: float | None = None,
        component: str = "intensity",
    ) -> torch.Tensor:
        flat_points, flat_viewdirs = self._flatten(points, viewdirs)
        component = str(component).lower()
        if component == "intensity":
            output = model.query(flat_points, flat_viewdirs, alpha=alpha)
        elif component in {"anatomy", "speckle"}:
            output = model.query_components(
                flat_points,
                flat_viewdirs,
                alpha=alpha,
            )[component]
        else:
            raise ValueError("component must be intensity, anatomy, or speckle")
        return output.to(DEVICE)

    def query_point_components(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        *,
        alpha: float | None = None,
    ) -> dict[str, torch.Tensor]:
        flat_points, flat_viewdirs = self._flatten(points, viewdirs)
        return {
            name: value.to(DEVICE)
            for name, value in model.query_components(
                flat_points,
                flat_viewdirs,
                alpha=alpha,
            ).items()
        }

    def query_points_masked(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        alpha: float | None = None,
        component: str = "intensity",
    ) -> torch.Tensor:
        flat_points, flat_viewdirs = self._flatten(points, viewdirs)
        flat_mask = torch.as_tensor(
            valid_mask,
            dtype=torch.bool,
            device=flat_points.device,
        ).reshape(-1)
        if flat_mask.numel() != flat_points.shape[0]:
            raise ValueError("valid_mask must contain one value per query point")
        valid_output = self.query_points(
            model,
            flat_points[flat_mask],
            flat_viewdirs[flat_mask],
            alpha=alpha,
            component=component,
        )
        output = torch.zeros(
            (flat_points.shape[0], valid_output.shape[-1]),
            dtype=valid_output.dtype,
            device=valid_output.device,
        )
        output[flat_mask] = valid_output
        return output

    def render_slice_from_dataset_valid(
        self,
        model,
        slice_number: int,
        reshaped: bool = False,
        *,
        alpha: float | None = None,
        component: str = "intensity",
    ) -> torch.Tensor:
        points = self.dataset.get_slice_valid_points(slice_number)
        output = self.query_points_masked(
            model,
            points,
            self.dataset.get_slice_valid_viewdirs(slice_number),
            self.dataset.get_sector_mask(flatten=True, device=points.device),
            alpha=alpha,
            component=component,
        )
        if reshaped:
            return output.reshape(self.height_px, self.width_px)
        return output
