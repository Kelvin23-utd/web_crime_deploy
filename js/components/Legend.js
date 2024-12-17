// Add to CrimeMap namespace
window.CrimeMap = window.CrimeMap || {};

CrimeMap.Legend = class Legend {
    static addLegend(map, title) {
        const legend = L.control({position: 'bottomright'});
        legend.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `<h4>${title}</h4>` +
                `<i style="background: ${CrimeMap.CONFIG.HEATMAP_COLORS.HIGH}"></i> High (${CrimeMap.CONFIG.CRIME_THRESHOLDS.HIGH}+)<br>` +
                `<i style="background: ${CrimeMap.CONFIG.HEATMAP_COLORS.MEDIUM}"></i> Medium (${CrimeMap.CONFIG.CRIME_THRESHOLDS.MEDIUM}-${CrimeMap.CONFIG.CRIME_THRESHOLDS.HIGH-1})<br>` +
                `<i style="background: ${CrimeMap.CONFIG.HEATMAP_COLORS.LOW}"></i> Low (1-${CrimeMap.CONFIG.CRIME_THRESHOLDS.MEDIUM-1})<br>` +
                `<i style="background: ${CrimeMap.CONFIG.HEATMAP_COLORS.NONE}"></i> No Crime Data<br>` +
                '<div style="margin-top: 10px;">' +
                '<i style="background: none; border: 2px dashed #2563eb"></i>Dallas City Boundary</div>';
            return div;
        };
        legend.addTo(map);
        return legend;
    }
};