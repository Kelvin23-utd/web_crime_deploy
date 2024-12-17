// Add to CrimeMap namespace
window.CrimeMap = window.CrimeMap || {};

CrimeMap.App = class App {
    constructor() {
        this.mapManager = new CrimeMap.MapManager('map1', 'map2');
    }

    async initialize() {
        await this.mapManager.loadAndDisplayData();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Add any global event listeners here
        window.addEventListener('resize', () => {
            this.mapManager.map1.invalidateSize();
            this.mapManager.map2.invalidateSize();
        });

        // Example: Add refresh button functionality if needed
        // const refreshButton = document.getElementById('refresh-btn');
        // if (refreshButton) {
        //     refreshButton.addEventListener('click', () => {
        //         this.mapManager.refreshData();
        //     });
        // }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new CrimeMap.App();
    app.initialize().catch(error => {
        console.error('Error initializing application:', error);
        alert('Failed to initialize application. Please check console for details.');
    });
});