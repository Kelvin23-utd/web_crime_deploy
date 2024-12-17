// Create a namespace for our application
window.CrimeMap = window.CrimeMap || {};

CrimeMap.Utils = {
    calculateBounds(polygon) {
        return polygon.reduce((acc, point) => ({
            north: Math.max(acc.north, point[0]),
            south: Math.min(acc.south, point[0]),
            east: Math.max(acc.east, point[1]),
            west: Math.min(acc.west, point[1])
        }), {north: -90, south: 90, east: -180, west: 180});
    },

    getHeatmapColor(count) {
        if (count === 0) return CrimeMap.CONFIG.HEATMAP_COLORS.NONE;
        if (count >= CrimeMap.CONFIG.CRIME_THRESHOLDS.HIGH) return CrimeMap.CONFIG.HEATMAP_COLORS.HIGH;
        if (count >= CrimeMap.CONFIG.CRIME_THRESHOLDS.MEDIUM) return CrimeMap.CONFIG.HEATMAP_COLORS.MEDIUM;
        return CrimeMap.CONFIG.HEATMAP_COLORS.LOW;
    },

    async loadMapData() {
        try {
            const [realResponse, predictedResponse] = await Promise.all([
                fetch(CrimeMap.CONFIG.DATA_URLS.REAL),
                fetch(CrimeMap.CONFIG.DATA_URLS.PREDICTED)
            ]);
            
            const realData = await realResponse.json();
            const predictedData = await predictedResponse.json();
            
            console.log("Real data format:", realData);
            console.log("Predicted data format:", predictedData);
            
            return { realData, predictedData };
        } catch (error) {
            console.error('Error loading data:', error);
            throw error;
        }
    }
};