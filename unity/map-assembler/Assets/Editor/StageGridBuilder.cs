using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace MaaDuelChannel.MapAssembler
{
    public sealed class GridCellRecord
    {
        public StageCellInput Cell;
        public GameObject Instance;
        public Vector3 WorldPosition;
    }

    public sealed class GridBuildResult
    {
        public GameObject Root;
        public Bounds WorldBounds;
        public List<GridCellRecord> Cells;
        public float ColumnPitch;
        public float RowPitch;
        public int RowCount;
        public int ColumnCount;
    }

    public static class StageGridBuilder
    {
        public static GridBuildResult Build(MapAssemblyInput input, BundleSet bundles)
        {
            if (input == null || input.grid == null || input.grid.Length == 0)
            {
                throw new InvalidDataException("Cannot build an empty stage grid.");
            }

            if (input.worldGrid == null || input.worldGrid.origin == null || input.worldGrid.origin.Length != 3)
            {
                throw new InvalidDataException("Assembly input has no world-grid transform.");
            }

            Dictionary<string, TilePrefabInput> prefabByKey = new Dictionary<string, TilePrefabInput>(StringComparer.Ordinal);
            foreach (TilePrefabInput mapping in input.tilePrefabByKey)
            {
                if (mapping == null || string.IsNullOrWhiteSpace(mapping.tileKey) || string.IsNullOrWhiteSpace(mapping.bundlePath)
                    || string.IsNullOrWhiteSpace(mapping.assetPath))
                {
                    throw new InvalidDataException("Assembly input contains an incomplete tile prefab mapping.");
                }

                if (prefabByKey.ContainsKey(mapping.tileKey))
                {
                    throw new InvalidDataException("Duplicate tile prefab mapping for key: " + mapping.tileKey);
                }

                prefabByKey.Add(mapping.tileKey, mapping);
            }

            int rowCount = 0;
            int columnCount = 0;
            HashSet<long> occupiedCoordinates = new HashSet<long>();
            foreach (StageCellInput cell in input.grid)
            {
                if (cell == null || cell.row < 0 || cell.column < 0 || string.IsNullOrWhiteSpace(cell.tileKey))
                {
                    throw new InvalidDataException("Assembly input contains an invalid stage cell.");
                }

                long key = ((long)cell.row << 32) | (uint)cell.column;
                if (!occupiedCoordinates.Add(key))
                {
                    throw new InvalidDataException("Stage grid contains duplicate row/column coordinates.");
                }

                if (!prefabByKey.ContainsKey(cell.tileKey))
                {
                    throw new InvalidDataException("No exact tile prefab mapping for stage tile key: " + cell.tileKey);
                }

                rowCount = Mathf.Max(rowCount, cell.row + 1);
                columnCount = Mathf.Max(columnCount, cell.column + 1);
            }

            if (rowCount * columnCount != input.grid.Length)
            {
                throw new InvalidDataException("Stage grid is not a complete rectangle of unique row/column cells.");
            }

            List<GridCellRecord> cells = new List<GridCellRecord>(input.grid.Length);
            List<StageCellInput> starts = new List<StageCellInput>();
            List<StageCellInput> ends = new List<StageCellInput>();
            GameObject root = new GameObject("StageGrid_" + input.stageKey.Replace('/', '_'));
            foreach (StageCellInput cell in input.grid)
            {
                TilePrefabInput mapping = prefabByKey[cell.tileKey];
                GameObject prefab = bundles.LoadGameObject(mapping.bundlePath, mapping.assetPath);
                Vector3 position = GridPosition(input, cell);
                GameObject instance = UnityEngine.Object.Instantiate(
                    prefab,
                    position,
                    prefab.transform.rotation,
                    root.transform
                );
                instance.name = "Cell_r" + cell.row.ToString("D2") + "_c" + cell.column.ToString("D2") + "_" + cell.tileKey;
                instance.transform.localScale = prefab.transform.localScale;
                cells.Add(new GridCellRecord { Cell = cell, Instance = instance, WorldPosition = instance.transform.position });

                if (cell.tileKey == "tile_start_dqq")
                {
                    starts.Add(cell);
                }
                else if (cell.tileKey == "tile_end_dqq")
                {
                    ends.Add(cell);
                }
            }

            ValidateEndpointColumn(starts, 1, "start");
            ValidateEndpointColumn(ends, 13, "end");

            Vector3 center = GridPosition(input, new StageCellInput
            {
                row = (rowCount - 1) / 2,
                column = (columnCount - 1) / 2,
            });
            Vector3 extent = new Vector3((columnCount - 1) * input.worldGrid.columnPitch, (rowCount - 1) * input.worldGrid.rowPitch, 0f);
            Bounds worldBounds = new Bounds(center, extent);
            worldBounds.Encapsulate(center + new Vector3(0f, 0f, input.camera.highlandHeight));
            Debug.Log(
                "Built " + cells.Count + " stage cell records (" + rowCount + "x" + columnCount + "), columnPitch="
                + input.worldGrid.columnPitch.ToString("F4") + ", rowPitch=" + input.worldGrid.rowPitch.ToString("F4")
                + ", origin=" + FormatVector(new Vector3(input.worldGrid.origin[0], input.worldGrid.origin[1], input.worldGrid.origin[2]))
                + ", transformEvidence=" + input.worldGrid.evidence + ", gridBounds=" + worldBounds
            );

            return new GridBuildResult
            {
                Root = root,
                WorldBounds = worldBounds,
                Cells = cells,
                ColumnPitch = input.worldGrid.columnPitch,
                RowPitch = input.worldGrid.rowPitch,
                RowCount = rowCount,
                ColumnCount = columnCount,
            };
        }

        private static Vector3 GridPosition(MapAssemblyInput input, StageCellInput cell)
        {
            float[] origin = input.worldGrid.origin;
            return new Vector3(
                origin[0] + (cell.column * input.worldGrid.columnPitch * input.worldGrid.columnDirection),
                origin[1] + (cell.row * input.worldGrid.rowPitch * input.worldGrid.rowDirection),
                origin[2] + (cell.heightType * input.camera.highlandHeight)
            );
        }

        private static void ValidateEndpointColumn(List<StageCellInput> cells, int expectedColumn, string label)
        {
            if (cells.Count != 9)
            {
                throw new InvalidDataException("Expected nine " + label + " tiles, found " + cells.Count + ".");
            }

            cells.Sort((left, right) => left.row.CompareTo(right.row));
            for (int index = 0; index < cells.Count; index++)
            {
                if (cells[index].row != index + 1 || cells[index].column != expectedColumn)
                {
                    throw new InvalidDataException(
                        "Unexpected " + label + " tile at row " + cells[index].row + ", column " + cells[index].column
                    );
                }
            }
        }

        private static string FormatVector(Vector3 value)
        {
            return "(" + value.x.ToString("F4") + ", " + value.y.ToString("F4") + ", " + value.z.ToString("F4") + ")";
        }
    }
}
