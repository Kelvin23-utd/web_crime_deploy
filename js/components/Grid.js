// Add to CrimeMap namespace
window.CrimeMap = window.CrimeMap || {};

CrimeMap.Grid = class Grid {
    constructor(bounds) {
        this.bounds = bounds;
    }

    createGrid(data) {
        const grid = {};
        
        for (let lat = this.bounds.south; lat < this.bounds.north; lat += CrimeMap.CONFIG.CELL_SIZE_DEG) {
            for (let lon = this.bounds.west; lon < this.bounds.east; lon += CrimeMap.CONFIG.CELL_SIZE_DEG) {
                const key = `${lat.toFixed(6)},${lon.toFixed(6)}`;
                grid[key] = {
                    lat: lat,
                    lon: lon,
                    count: 0
                };
            }
        }

        const predictions = data.predictions || [];
        predictions.forEach(pred => {
            let lat, lon, crimeCount;
            
            if (Array.isArray(pred)) {
                [lat, lon] = pred;
                crimeCount = 1;
            } else {
                lat = pred.latitude;
                lon = pred.longitude;
                crimeCount = pred.predicted_crimes || 1;
            }

            const latIndex = Math.floor((lat - this.bounds.south) / CrimeMap.CONFIG.CELL_SIZE_DEG);
            const lonIndex = Math.floor((lon - this.bounds.west) / CrimeMap.CONFIG.CELL_SIZE_DEG);
            const cellLat = this.bounds.south + (latIndex * CrimeMap.CONFIG.CELL_SIZE_DEG);
            const cellLon = this.bounds.west + (lonIndex * CrimeMap.CONFIG.CELL_SIZE_DEG);
            const key = `${cellLat.toFixed(6)},${cellLon.toFixed(6)}`;
            
            if (grid[key]) {
                grid[key].count += crimeCount;
            }
        });

        return grid;
    }

    addGridToMap(grid, map, isActualData = true) {
        const gridLayer = L.featureGroup();
        
        Object.values(grid).forEach(cell => {
            const bounds = [
                [cell.lat, cell.lon],
                [cell.lat + CrimeMap.CONFIG.CELL_SIZE_DEG, cell.lon + CrimeMap.CONFIG.CELL_SIZE_DEG]
            ];
            
            const rectangle = L.rectangle(bounds, {
                color: 'transparent',
                weight: 0,
                fillColor: CrimeMap.Utils.getHeatmapColor(cell.count),
                fillOpacity: cell.count > 0 ? 0.6 : 0.3,
                className: 'grid-cell'
            });

            rectangle.bindPopup(`${isActualData ? 'Actual' : 'Predicted'} Crimes: ${cell.count}`);
            
            gridLayer.addLayer(rectangle);
            
            if (!map._gridCells) map._gridCells = {};
            map._gridCells[`${cell.lat},${cell.lon}`] = rectangle;
        });
        
        gridLayer.addTo(map);
        return gridLayer;
    }
};