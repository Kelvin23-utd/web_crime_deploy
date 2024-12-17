// Create a namespace for our application
window.CrimeMap = window.CrimeMap || {};

CrimeMap.CONFIG = {
    // Dallas polygon coordinates
    DALLAS_POLYGON: [
        [32.869839, -96.936899],
        [32.977145, -96.837507],
        [32.906461, -96.733480],
        [32.910587, -96.642199],
        [32.859866, -96.587225],
        [32.712301, -96.610572],
        [32.640769, -96.662670],
        [32.640769, -96.836477],
        [32.774660, -96.991724],
        [32.869839, -96.936899]
    ],
    
    CELL_SIZE_DEG: 0.014492,
    
    MAP_CENTER: [32.7767, -96.7970],
    MAP_ZOOM: 11,
    
    HEATMAP_COLORS: {
        NONE: '#e5f5e0',
        LOW: '#ffff00',
        MEDIUM: '#ff8800',
        HIGH: '#ff0000'
    },
    
    CRIME_THRESHOLDS: {
        HIGH: 8,
        MEDIUM: 4
    },
    
    DATA_URLS: {
        REAL: './real.json',
        PREDICTED: './output/crime_predictions.json'
    }
};