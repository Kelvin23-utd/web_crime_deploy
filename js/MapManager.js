// Add to CrimeMap namespace
window.CrimeMap = window.CrimeMap || {};

CrimeMap.MapManager = class MapManager {
    constructor(map1Id, map2Id) {
        this.map1 = L.map(map1Id).setView(CrimeMap.CONFIG.MAP_CENTER, CrimeMap.CONFIG.MAP_ZOOM);
        this.map2 = L.map(map2Id).setView(CrimeMap.CONFIG.MAP_CENTER, CrimeMap.CONFIG.MAP_ZOOM);
        this.bounds = CrimeMap.Utils.calculateBounds(CrimeMap.CONFIG.DALLAS_POLYGON);
        this.grid = new CrimeMap.Grid(this.bounds);
        
        this.initialize();
    }

    initialize() {
        this.setupTileLayers();
        this.syncMaps();
    }

    setupTileLayers() {
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map1);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map2);
    }

    syncMaps() {
        this.syncMapMovement(this.map1, this.map2);
        this.syncMapMovement(this.map2, this.map1);
    }

    syncMapMovement(sourceMap, targetMap) {
        sourceMap.on('move', () => {
            targetMap.setView(sourceMap.getCenter(), sourceMap.getZoom(), {
                animate: false
            });
        });
        
        sourceMap.on('zoom', () => {
            targetMap.setZoom(sourceMap.getZoom(), {
                animate: false
            });
        });
    }

    async loadAndDisplayData() {
        try {
            const { realData, predictedData } = await CrimeMap.Utils.loadMapData();
            
            // Create grids for both datasets
            const realGrid = this.grid.createGrid(realData);
            const predictedGrid = this.grid.createGrid(predictedData);
            
            // Add grids to maps
            const gridLayer1 = this.grid.addGridToMap(realGrid, this.map1, true);
            const gridLayer2 = this.grid.addGridToMap(predictedGrid, this.map2, false);

            // Add Dallas boundaries
            const boundary1 = CrimeMap.Boundary.addDallasBoundary(this.map1);
            const boundary2 = CrimeMap.Boundary.addDallasBoundary(this.map2);

            // Add legends
            CrimeMap.Legend.addLegend(this.map1, 'Actual Crime Density');
            CrimeMap.Legend.addLegend(this.map2, 'Predicted Crime Density');

            // Fit to Dallas boundary
            const polyBounds = L.latLngBounds(CrimeMap.CONFIG.DALLAS_POLYGON);
            this.map1.fitBounds(polyBounds);

            // Ensure boundaries stay on top
            boundary1.bringToFront();
            boundary2.bringToFront();

            // Store references for later use
            this.layers = {
                realGrid: gridLayer1,
                predictedGrid: gridLayer2,
                boundary1,
                boundary2
            };

        } catch (error) {
            console.error('Error initializing maps:', error);
            alert('Error loading data. Check console for details.');
        }
    }

    async refreshData() {
        // Remove existing layers
        if (this.layers) {
            this.layers.realGrid.remove();
            this.layers.predictedGrid.remove();
            this.layers.boundary1.remove();
            this.layers.boundary2.remove();
        }

        // Reload data
        await this.loadAndDisplayData();
    }
};