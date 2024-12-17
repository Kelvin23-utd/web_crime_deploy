// Add to CrimeMap namespace
window.CrimeMap = window.CrimeMap || {};

CrimeMap.Boundary = class Boundary {
    static addDallasBoundary(map) {
        return L.polygon(CrimeMap.CONFIG.DALLAS_POLYGON, {
            color: '#2563eb',
            weight: 3,
            fill: false,
            dashArray: '5, 10'
        }).addTo(map);
    }
};