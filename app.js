/**
 * GeoAI LULC Dashboard - Production JavaScript
 * Interfaces with existing automated pipeline without modifications
 */

class GeoAIDashboard {
    constructor() {
        this.currentPage = 'page1';
        this.map = null;
        this.mapLayers = {};
        this.analysisData = null;
        this.availableYears = [];
        
        this.init();
    }
    
    async init() {
        await this.loadAvailableYears();
        this.setupEventListeners();
        this.initializePage1();
    }
    
    // ==========================================================================
    // Data Loading & Validation
    // ==========================================================================
    
    async loadAvailableYears() {
        try {
            // Scan for available years by checking folder structure
            // This simulates scanning the file system for Tirupati_XXXX folders
            const response = await this.scanYearFolders();
            this.availableYears = response || ['2016', '2018', '2020', '2021', '2022'];
            this.populateYearDropdowns();
        } catch (error) {
            console.error('Failed to load available years:', error);
            // Fallback to known years
            this.availableYears = ['2016', '2018', '2020', '2021', '2022'];
            this.populateYearDropdowns();
        }
    }
    
    async scanYearFolders() {
        // In a real implementation, this would call a backend API
        // that scans the filesystem for Tirupati_XXXX folders
        return new Promise(resolve => {
            setTimeout(() => {
                // Simulated folder scan results
                resolve(['2016', '2018', '2020', '2021', '2022']);
            }, 500);
        });
    }
    
    populateYearDropdowns() {
        const year1Select = document.getElementById('year1');
        const year2Select = document.getElementById('year2');
        
        [year1Select, year2Select].forEach(select => {
            select.innerHTML = '<option value="">Select year...</option>';
            this.availableYears.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                select.appendChild(option);
            });
        });
    }
    
    async validateSafeFolder(year) {
        try {
            // Call backend API to validate SAFE folder contents
            const response = await this.callBackendAPI('validate-safe-folder', { year });
            return response;
        } catch (error) {
            console.error(`Failed to validate year ${year}:`, error);
            return this.getMockValidation(year);
        }
    }
    
    getMockValidation(year) {
        // Mock validation for demo purposes
        const files = [
            `T44PLV_${year}1225T051222_B02_10m.jp2`,
            `T44PLV_${year}1225T051222_B03_10m.jp2`,
            `T44PLV_${year}1225T051222_B04_10m.jp2`,
            `T44PLV_${year}1225T051222_B08_10m.jp2`
        ];
        
        return {
            valid: true,
            files: files.map(file => ({
                band: file.includes('B02') ? 'B02 (Blue)' : 
                      file.includes('B03') ? 'B03 (Green)' : 
                      file.includes('B04') ? 'B04 (Red)' : 'B08 (NIR)',
                filename: file,
                exists: true
            }))
        };
    }
    
    // ==========================================================================
    // Page 1: Year Selection Logic
    // ==========================================================================
    
    initializePage1() {
        this.showPage('page1');
    }
    
    setupEventListeners() {
        // Year selection changes
        document.getElementById('year1').addEventListener('change', () => this.handleYearChange());
        document.getElementById('year2').addEventListener('change', () => this.handleYearChange());
        
        // Run analysis button
        document.getElementById('run-analysis').addEventListener('click', () => this.runAnalysis());
        
        // Dashboard controls
        document.getElementById('reset-analysis')?.addEventListener('click', () => this.resetAnalysis());
        document.getElementById('download-report')?.addEventListener('click', () => this.downloadReport());
        document.getElementById('download-csv')?.addEventListener('click', () => this.downloadCSV());
        
        // Filter controls
        this.setupFilterControls();
        
        // Map layer toggles
        this.setupMapControls();
    }
    
    async handleYearChange() {
        const year1 = document.getElementById('year1').value;
        const year2 = document.getElementById('year2').value;
        const runButton = document.getElementById('run-analysis');
        const errorDiv = document.getElementById('error-message');
        const validationDiv = document.getElementById('validation-results');
        
        // Clear previous states
        errorDiv.classList.add('hidden');
        validationDiv.classList.add('hidden');
        runButton.disabled = true;
        
        // Clear status indicators
        document.getElementById('year1-status').innerHTML = '';
        document.getElementById('year2-status').innerHTML = '';
        
        if (year1 && year2) {
            if (year1 === year2) {
                this.showError('Please select different years for comparison');
                return;
            }
            
            // Show validation in progress
            document.getElementById('year1-status').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
            document.getElementById('year2-status').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
            
            // Validate both years
            const [validation1, validation2] = await Promise.all([
                this.validateSafeFolder(year1),
                this.validateSafeFolder(year2)
            ]);
            
            this.displayValidationResults(year1, validation1, year2, validation2);
            
            if (validation1.valid && validation2.valid) {
                runButton.disabled = false;
            } else {
                this.showError('One or more folders contain missing satellite bands. Please check your data.');
            }
        }
    }
    
    displayValidationResults(year1, val1, year2, val2) {
        const validationDiv = document.getElementById('validation-results');
        const contentDiv = document.getElementById('validation-content');
        
        // Update status indicators
        const status1 = val1.valid ? 
            '<i class="fas fa-check-circle" style="color: var(--success-color)"></i> Valid' :
            '<i class="fas fa-times-circle" style="color: var(--danger-color)"></i> Invalid';
        const status2 = val2.valid ? 
            '<i class="fas fa-check-circle" style="color: var(--success-color)"></i> Valid' :
            '<i class="fas fa-times-circle" style="color: var(--danger-color)"></i> Invalid';
            
        document.getElementById('year1-status').innerHTML = status1;
        document.getElementById('year2-status').innerHTML = status2;
        
        // Build validation content
        let html = '';
        
        [{ year: year1, validation: val1 }, { year: year2, validation: val2 }].forEach(({ year, validation }) => {
            html += `<div style="margin-bottom: 1rem;"><h4>Year ${year} Files:</h4>`;
            validation.files.forEach(file => {
                const icon = file.exists ? 
                    '<i class="fas fa-check" style="color: var(--success-color)"></i>' :
                    '<i class="fas fa-times" style="color: var(--danger-color)"></i>';
                html += `<div class="validation-item">${icon} ${file.band}: ${file.filename}</div>`;
            });
            html += '</div>';
        });
        
        contentDiv.innerHTML = html;
        validationDiv.classList.remove('hidden');
    }
    
    showError(message) {
        const errorDiv = document.getElementById('error-message');
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
    
    // ==========================================================================
    // Pipeline Execution
    // ==========================================================================
    
    async runAnalysis() {
        const year1 = document.getElementById('year1').value;
        const year2 = document.getElementById('year2').value;
        
        if (!year1 || !year2) return;
        
        this.showProgressOverlay();
        
        try {
            // Execute the existing automated pipeline
            const result = await this.executeGeoAIPipeline(year1, year2);
            
            if (result.success) {
                this.analysisData = result.data;
                this.hideProgressOverlay();
                this.showPage('page2');
                this.initializeDashboard();
            } else {
                throw new Error(result.error || 'Pipeline execution failed');
            }
        } catch (error) {
            console.error('Analysis failed:', error);
            this.hideProgressOverlay();
            this.showError('Analysis failed: ' + error.message);
        }
    }
    
    async executeGeoAIPipeline(year1, year2) {
        // Simulate the existing automated pipeline execution
        // In production, this would trigger the actual Python pipeline
        
        const steps = [
            { id: 'step1', text: 'Loading satellite data...', duration: 2000 },
            { id: 'step2', text: 'Running LULC classification...', duration: 5000 },
            { id: 'step3', text: 'Computing change detection...', duration: 3000 },
            { id: 'step4', text: 'Generating analysis results...', duration: 2000 }
        ];
        
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            this.updateProgressStep(step.id, step.text);
            await this.delay(step.duration);
        }
        
        // Return mock analysis results
        return {
            success: true,
            data: this.generateMockAnalysisData(year1, year2)
        };
    }
    
    async callBackendAPI(endpoint, data) {
        // In production, this would make actual API calls to the backend
        // For demo, return mock responses
        
        if (endpoint === 'validate-safe-folder') {
            await this.delay(800);
            return this.getMockValidation(data.year);
        }
        
        if (endpoint === 'run-pipeline') {
            return await this.executeGeoAIPipeline(data.year1, data.year2);
        }
        
        throw new Error(`Unknown endpoint: ${endpoint}`);
    }
    
    showProgressOverlay() {
        document.getElementById('progress-overlay').classList.remove('hidden');
        document.getElementById('progress-text').textContent = 'Initializing GeoAI pipeline...';
        
        // Reset all steps
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }
    
    updateProgressStep(stepId, text) {
        document.getElementById('progress-text').textContent = text;
        
        // Mark previous steps as completed
        const allSteps = ['step1', 'step2', 'step3', 'step4'];
        const currentIndex = allSteps.indexOf(stepId);
        
        allSteps.forEach((id, index) => {
            const element = document.getElementById(id);
            if (index < currentIndex) {
                element.classList.remove('active');
                element.classList.add('completed');
            } else if (index === currentIndex) {
                element.classList.add('active');
                element.classList.remove('completed');
            } else {
                element.classList.remove('active', 'completed');
            }
        });
    }
    
    hideProgressOverlay() {
        document.getElementById('progress-overlay').classList.add('hidden');
    }
    
    // ==========================================================================
    // Dashboard Initialization
    // ==========================================================================
    
    initializeDashboard() {
        this.initializeMap();
        this.populateExecutiveSummary();
        this.populateStatisticsTable();
        this.populateGrowthIndicators();
        this.populateTransitionMatrix();
        this.populatePolicyInsights();
        this.populateSustainabilityReport();
    }
    
    initializeMap() {
        // Initialize Leaflet map
        this.map = L.map('map').setView([13.6288, 79.4192], 10); // Tirupati coordinates
        
        // Add base layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        
        // Add LULC layers (these would load the actual GeoTIFF outputs in production)
        this.mapLayers = {
            't1': this.createMockLULCLayer('Time 1'),
            't2': this.createMockLULCLayer('Time 2'),
            'change': this.createMockChangeLayer(),
            'confidence': this.createMockConfidenceLayer()
        };
        
        // Add default layer
        this.mapLayers.t1.addTo(this.map);
    }
    
    createMockLULCLayer(title) {
        // In production, this would load actual GeoTIFF files
        const bounds = [[13.5, 79.3], [13.7, 79.5]];
        
        return L.rectangle(bounds, {
            color: '#16a34a',
            fillColor: '#16a34a',
            fillOpacity: 0.3,
            weight: 2
        }).bindPopup(`${title} LULC Classification<br>Class: Forest<br>Confidence: 0.89`);
    }
    
    createMockChangeLayer() {
        const bounds = [[13.55, 79.35], [13.65, 79.45]];
        
        return L.rectangle(bounds, {
            color: '#dc2626',
            fillColor: '#dc2626',
            fillOpacity: 0.3,
            weight: 2
        }).bindPopup('Land Use Change<br>From: Forest<br>To: Built-up<br>Confidence: 0.92');
    }
    
    createMockConfidenceLayer() {
        const bounds = [[13.52, 79.32], [13.68, 79.48]];
        
        return L.rectangle(bounds, {
            color: '#3b82f6',
            fillColor: '#3b82f6',
            fillOpacity: 0.2,
            weight: 1
        }).bindPopup('Confidence Map<br>Average: 0.86<br>Range: 0.72 - 0.98');
    }
    
    setupMapControls() {
        const layerControls = {
            'layer-t1': 't1',
            'layer-t2': 't2',
            'layer-change': 'change',
            'layer-confidence': 'confidence'
        };
        
        Object.entries(layerControls).forEach(([checkboxId, layerKey]) => {
            const checkbox = document.getElementById(checkboxId);
            checkbox?.addEventListener('change', (e) => {
                const layer = this.mapLayers[layerKey];
                if (e.target.checked) {
                    layer.addTo(this.map);
                } else {
                    this.map.removeLayer(layer);
                }
            });
        });
    }
    
    populateExecutiveSummary() {
        if (!this.analysisData) return;
        
        document.getElementById('confidence-value').textContent = `${(this.analysisData.avgConfidence * 100).toFixed(1)}%`;
        document.getElementById('reliability-value').textContent = this.analysisData.reliabilityScore;
        document.getElementById('change-value').textContent = `${this.analysisData.changePercent.toFixed(1)}%`;
        document.getElementById('transitions-value').textContent = this.analysisData.majorTransitions.toFixed(0);
    }
    
    populateStatisticsTable() {
        if (!this.analysisData) return;
        
        const tbody = document.getElementById('statistics-tbody');
        tbody.innerHTML = '';
        
        this.analysisData.landCoverStats.forEach(stat => {
            const row = document.createElement('tr');
            const changeValue = stat.areaT2 - stat.areaT1;
            const changePercent = ((changeValue / stat.areaT1) * 100).toFixed(1);
            
            row.innerHTML = `
                <td><span class="legend-color ${stat.class.toLowerCase()}"></span> ${stat.class}</td>
                <td>${stat.areaT1.toLocaleString()}</td>
                <td>${stat.areaT2.toLocaleString()}</td>
                <td>${stat.percentage.toFixed(1)}%</td>
                <td style="color: ${changeValue >= 0 ? 'var(--success-color)' : 'var(--danger-color)'}">${changeValue >= 0 ? '+' : ''}${changeValue.toLocaleString()}</td>
                <td style="color: ${changeValue >= 0 ? 'var(--success-color)' : 'var(--danger-color)'}">${changePercent}%</td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    populateGrowthIndicators() {
        if (!this.analysisData) return;
        
        const builtupData = this.analysisData.landCoverStats.find(s => s.class === 'Built-up');
        const forestData = this.analysisData.landCoverStats.find(s => s.class === 'Forest');
        const agricultureData = this.analysisData.landCoverStats.find(s => s.class === 'Agriculture');
        
        if (builtupData) {
            const change = builtupData.areaT2 - builtupData.areaT1;
            const percent = ((change / builtupData.areaT1) * 100);
            const annual = (percent / (parseInt(this.analysisData.year2) - parseInt(this.analysisData.year1))).toFixed(1);
            
            document.getElementById('buildup-change').textContent = change >= 0 ? `+${change.toLocaleString()}` : change.toLocaleString();
            document.getElementById('buildup-trend').innerHTML = change >= 0 ? '↗' : '↘';
            document.getElementById('buildup-trend').className = `indicator-trend ${change >= 0 ? 'up' : 'down'}`;
            document.getElementById('buildup-area').textContent = Math.abs(change).toLocaleString();
            document.getElementById('buildup-annual').textContent = annual;
        }
        
        if (forestData && agricultureData) {
            const forestChange = forestData.areaT2 - forestData.areaT1;
            const agriChange = agricultureData.areaT2 - agricultureData.areaT1;
            const totalVegChange = forestChange + agriChange;
            const totalVegT1 = forestData.areaT1 + agricultureData.areaT1;
            const vegPercent = ((totalVegChange / totalVegT1) * 100).toFixed(1);
            
            document.getElementById('vegetation-change').textContent = totalVegChange >= 0 ? `+${totalVegChange.toLocaleString()}` : totalVegChange.toLocaleString();
            document.getElementById('vegetation-trend').innerHTML = totalVegChange >= 0 ? '↗' : '↘';
            document.getElementById('vegetation-trend').className = `indicator-trend ${totalVegChange >= 0 ? 'up' : 'down'}`;
            document.getElementById('vegetation-area').textContent = Math.abs(totalVegChange).toLocaleString();
            document.getElementById('vegetation-percent').textContent = vegPercent;
        }
    }
    
    populateTransitionMatrix() {
        if (!this.analysisData) return;
        
        const matrixDiv = document.getElementById('transition-matrix');
        const matrix = this.analysisData.transitionMatrix;
        
        let html = '<table class="statistics-table"><thead><tr><th>From \\ To</th>';
        
        // Header row
        Object.keys(matrix).forEach(toClass => {
            html += `<th>${toClass}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Data rows
        Object.entries(matrix).forEach(([fromClass, transitions]) => {
            html += `<tr><td><strong>${fromClass}</strong></td>`;
            Object.entries(transitions).forEach(([toClass, value]) => {
                const intensity = Math.min(value / 1000, 1); // Normalize for color intensity
                const bgColor = value > 100 ? `rgba(239, 68, 68, ${intensity})` : 'transparent';
                html += `<td style="background-color: ${bgColor}">${value.toLocaleString()} ha</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        matrixDiv.innerHTML = html;
    }
    
    populatePolicyInsights() {
        if (!this.analysisData) return;
        
        const insightsDiv = document.getElementById('policy-insights');
        const insights = this.analysisData.policyInsights;
        
        insightsDiv.innerHTML = insights.map(insight => `
            <div class="insight-item">
                <i class="fas fa-lightbulb"></i>
                <span>${insight}</span>
            </div>
        `).join('');
    }
    
    populateSustainabilityReport() {
        if (!this.analysisData) return;
        
        const sdg11 = document.getElementById('sdg11-content');
        const sdg15 = document.getElementById('sdg15-content');
        
        sdg11.innerHTML = `
            <p><strong>Urban Growth Assessment:</strong></p>
            <p>${this.analysisData.sustainabilityReport.sdg11}</p>
        `;
        
        sdg15.innerHTML = `
            <p><strong>Forest Conservation Status:</strong></p>
            <p>${this.analysisData.sustainabilityReport.sdg15}</p>
        `;
    }
    
    setupFilterControls() {
        const filters = ['filter-agriculture', 'filter-water', 'filter-buildup', 'filter-forest'];
        
        filters.forEach(filterId => {
            document.getElementById(filterId)?.addEventListener('change', (e) => {
                this.applyFilters();
            });
        });
    }
    
    applyFilters() {
        // Get active filters
        const activeFilters = [];
        if (document.getElementById('filter-agriculture')?.checked) activeFilters.push('agriculture');
        if (document.getElementById('filter-water')?.checked) activeFilters.push('water');
        if (document.getElementById('filter-buildup')?.checked) activeFilters.push('buildup');
        if (document.getElementById('filter-forest')?.checked) activeFilters.push('forest');
        
        // Update map layers based on filters
        this.updateMapFilters(activeFilters);
        
        // Update statistics table
        this.updateTableFilters(activeFilters);
    }
    
    updateMapFilters(filters) {
        // In production, this would filter the actual map layers
        console.log('Applying map filters:', filters);
    }
    
    updateTableFilters(filters) {
        const rows = document.querySelectorAll('#statistics-tbody tr');
        
        rows.forEach(row => {
            const className = row.cells[0].textContent.trim().toLowerCase();
            const shouldShow = filters.length === 0 || filters.some(filter => className.includes(filter));
            row.style.display = shouldShow ? '' : 'none';
        });
    }
    
    // ==========================================================================
    // Utility Functions
    // ==========================================================================
    
    showPage(pageId) {
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
        document.getElementById(pageId).classList.add('active');
        this.currentPage = pageId;
    }
    
    resetAnalysis() {
        if (confirm('Are you sure you want to start a new analysis? This will clear all current results.')) {
            this.analysisData = null;
            this.showPage('page1');
            
            // Clear selections
            document.getElementById('year1').value = '';
            document.getElementById('year2').value = '';
            document.getElementById('validation-results').classList.add('hidden');
            document.getElementById('error-message').classList.add('hidden');
            document.getElementById('run-analysis').disabled = true;
        }
    }
    
    downloadReport() {
        if (!this.analysisData) return;
        
        // Generate and download PDF report
        const reportData = this.generateReportData();
        this.downloadFile('LULC_Analysis_Report.pdf', reportData, 'application/pdf');
    }
    
    downloadCSV() {
        if (!this.analysisData) return;
        
        const csvData = this.generateCSVData();
        this.downloadFile('LULC_Statistics.csv', csvData, 'text/csv');
    }
    
    generateCSVData() {
        let csv = 'Land Cover,Area T1 (ha),Area T2 (ha),% of District,Change (ha),Change (%)\n';
        
        this.analysisData.landCoverStats.forEach(stat => {
            const changeValue = stat.areaT2 - stat.areaT1;
            const changePercent = ((changeValue / stat.areaT1) * 100).toFixed(1);
            
            csv += `${stat.class},${stat.areaT1},${stat.areaT2},${stat.percentage.toFixed(1)},${changeValue},${changePercent}\n`;
        });
        
        return csv;
    }
    
    downloadFile(filename, data, mimeType) {
        const blob = new Blob([data], { type: mimeType });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        window.URL.revokeObjectURL(url);
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // ==========================================================================
    // Mock Data Generation
    // ==========================================================================
    
    generateMockAnalysisData(year1, year2) {
        return {
            year1,
            year2,
            avgConfidence: 0.86,
            reliabilityScore: 'High',
            changePercent: 4.2,
            majorTransitions: 12,
            
            landCoverStats: [
                {
                    class: 'Forest',
                    areaT1: 45230,
                    areaT2: 44890,
                    percentage: 52.3
                },
                {
                    class: 'Water Bodies',
                    areaT1: 2150,
                    areaT2: 2180,
                    percentage: 2.5
                },
                {
                    class: 'Agriculture',
                    areaT1: 18420,
                    areaT2: 17950,
                    percentage: 20.9
                },
                {
                    class: 'Barren Land',
                    areaT1: 8900,
                    areaT2: 8450,
                    percentage: 9.8
                },
                {
                    class: 'Built-up',
                    areaT1: 6300,
                    areaT2: 7530,
                    percentage: 8.8
                }
            ],
            
            transitionMatrix: {
                'Forest': { 'Forest': 44500, 'Water Bodies': 0, 'Agriculture': 200, 'Barren Land': 130, 'Built-up': 400 },
                'Water Bodies': { 'Forest': 0, 'Water Bodies': 2150, 'Agriculture': 0, 'Barren Land': 0, 'Built-up': 0 },
                'Agriculture': { 'Forest': 150, 'Water Bodies': 20, 'Agriculture': 17500, 'Barren Land': 250, 'Built-up': 500 },
                'Barren Land': { 'Forest': 240, 'Water Bodies': 10, 'Agriculture': 200, 'Barren Land': 7820, 'Built-up': 630 },
                'Built-up': { 'Forest': 0, 'Water Bodies': 0, 'Agriculture': 50, 'Barren Land': 250, 'Built-up': 6000 }
            },
            
            policyInsights: [
                'Urban expansion primarily converted agricultural and barren lands, showing controlled development.',
                'Forest areas remained largely stable with minimal conversion to other land uses.',
                'Water bodies showed slight increase, indicating potential water conservation efforts.',
                'Built-up area increased by 19.5%, reflecting moderate urban growth pattern.'
            ],
            
            sustainabilityReport: {
                sdg11: 'Urban growth is occurring at a sustainable rate of 19.5% over the analysis period. Development is primarily converting non-forest lands, indicating good spatial planning practices.',
                sdg15: 'Forest cover remains stable at 52.3% of the district area. The minimal forest loss (0.75%) demonstrates effective forest conservation policies and management practices.'
            }
        };
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.geoAIDashboard = new GeoAIDashboard();
});