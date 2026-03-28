// === Navbar Scroll Effect ===
const navbar = document.getElementById('navbar');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
  if (window.scrollY > 50) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
  
  updateActiveNavLink();
});

// === Mobile Menu Toggle ===
const menuToggle = document.querySelector('.menu-toggle');
const navMenu = document.getElementById('nav-menu');

menuToggle.addEventListener('click', () => {
  navMenu.classList.toggle('show');
  document.body.classList.toggle('menu-open');
  
  if (navMenu.classList.contains('show')) {
    menuToggle.classList.remove('fa-bars');
    menuToggle.classList.add('fa-xmark');
  } else {
    menuToggle.classList.remove('fa-xmark');
    menuToggle.classList.add('fa-bars');
  }
});

navLinks.forEach(link => {
  link.addEventListener('click', () => {
    navMenu.classList.remove('show');
    document.body.classList.remove('menu-open');
    menuToggle.classList.remove('fa-xmark');
    menuToggle.classList.add('fa-bars');
  });
});

document.addEventListener('click', (e) => {
  if (navMenu.classList.contains('show') && 
      !navMenu.contains(e.target) && 
      !menuToggle.contains(e.target)) {
    navMenu.classList.remove('show');
    document.body.classList.remove('menu-open');
    menuToggle.classList.remove('fa-xmark');
    menuToggle.classList.add('fa-bars');
  }
});

// === Smooth Scroll with Active Nav Link ===
function updateActiveNavLink() {
  const sections = document.querySelectorAll('section[id]');
  const scrollY = window.pageYOffset;

  sections.forEach(section => {
    const sectionHeight = section.offsetHeight;
    const sectionTop = section.offsetTop - 100;
    const sectionId = section.getAttribute('id');
    const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);

    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
      navLinks.forEach(link => link.classList.remove('active'));
      if (navLink) navLink.classList.add('active');
    }
  });
}

// === PROJECT MODAL SYSTEM ===

// Project data structure
const projectsData = {
  'project1': {
    title: 'Differentiate Privacy on Voting Service (Fider)',
    category: 'Data Privacy',
    image: 'image/privacy.jpg',
    description: `This is a detailed description of your project. Explain what problem it solves, the approach you took, and the impact it had. Be specific about challenges you overcame and technologies you used.`,
    longDescription: `<strong><i>Problem:</i></strong><br>
Traditional voting systems expose exact vote counts in real-time, enabling observers to infer individual behavior through:
<br>• Timing attacks (watching counts change when someone votes)
<br>• Averaging attacks (querying repeatedly to remove noise)
<br>• Threshold crossing leaks (seeing when a count becomes visible)

<br><br><strong><i>Solution:</i></strong><br>
Built a differential privacy sidecar that:

<br>• Adds calibrated Laplace noise (ε=0.5) to vote counts
<br>• Releases results on fixed window timeframe (prevents timing attacks)
<br>• Reuses noise when counts unchanged (prevents averaging attacks)
<br>• Tracks privacy budget and lock the post when budget is exhaust.

<br><br><strong><i>Technical Architecture:</i></strong>
<br>• Frontend: Vanilla JavaScript displaying noisy counts with uncertainty ranges
<br>• Backend: FastAPI + PostgreSQL dual-database architecture
<br>• Privacy: ε-differential privacy with Laplace mechanism
<br>• Budget: Lifetime epsilon tracking with automatic post locking
<br>• Scheduler: APScheduler for batch publishing the noisy counts in a specific time`,
    tech: ['Python', 'FastAPI', 'Docker', 'SQL', 'Javascript', 'HTML/CSS'],
    features: [
      'Prevents timing attacks (fixed-schedule releases)',
      'Prevents averaging attacks (noise reuse)',
      'Budget tracking (lifetime epsilon cap)',
      'Post locking (final results after budget exhaustion)',
      'Individual votes protected with plausible deniability',
      'Balances utility with privacy'
    ],
    stats: [
      { value: '100%', label: 'Attack Prevention' },
      { value: '96%', label: 'Decision Accuracy' },
      { value: '98.3%', label: 'Privacy Budget Saving' }
    ],
    github: 'https://github.com/dangdinh99/fider-dp',
    demo: 'https://your-demo-link.com',
    report: 'files/dp_fider_report.pdf',
    hasGithub: true,
    hasDemo: true,
    hasReport: true 
  },
  'project2': {
    title: 'Campaign Zero: 911 Call Data',
    category: 'Data Pipeline | Data Engineer',
    image: 'image/campaignzero.png ',
    description: 'Scalable ETL pipeline harmonizing 60M+ 911 call records from 11+ cities into a unified Snowflake dataset for public safety analysis.',
    longDescription: `<div>
<h4 style="color: #00d4ff; margin-bottom: 12px; margin-top: 0;">Problem</h4>
<p style="margin-bottom: 16px;">911 call records are currently siloed in individual city data portals, often in inconsistent formats and lacking standardization. Each city uses different terminology, schemas, and classifications, making cross-city analysis nearly impossible.</p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li>No comprehensive multi-city dataset exists for comparative analysis</li>
  <li>Different cities use vastly different call type terminologies (e.g., Detroit: "Robbery armed ip-any" vs Seattle: "Robbery - armed")</li>
  <li>Data formats vary widely (CSV, Excel, inconsistent timestamps, missing geospatial data)</li>
  <li>NYC dataset alone exceeded 53 million records, requiring specialized processing</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Solution</h4>
<p style="margin-bottom: 20px;">Built a configuration-driven ETL pipeline that collects, standardizes, and harmonizes 911 call data from multiple major U.S. cities into a single Snowflake data warehouse, enabling meaningful cross-city comparisons and pattern analysis for Campaign Zero's public safety research.</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Technical Implementation</h4>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Architecture:</strong> 4-stage ETL pipeline (Ingestion → Preprocessing → Standardization → Output) controlled via YAML configuration files for scalability</li>
  <li><strong>Data Processing:</strong> Chunked processing for massive datasets (NYC: 53M records in 105 batches of 500K), optimized to run on 32GB RAM systems handling 20GB+ data</li>
  <li><strong>Standardization:</strong> Python mapping algorithm converting city-specific terminologies into unified 15-category taxonomy using keyword detection and pattern matching</li>
  <li><strong>Geographic Enrichment:</strong> Spatial joins with Census TIGER/Line shapefiles to assign GEOIDs (block groups) for demographic correlation</li>
  <li><strong>Storage Optimization:</strong> Parquet compression for efficient storage and SQL-based querying in Snowflake data warehouse</li>
  <li><strong>Extensibility:</strong> Configuration-driven design allowing new cities to be added with minimal code refactoring</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">My Contribution</h4>
<p style="margin-bottom: 20px;">As part of a 6-person team collaborating with BU Spark! and Campaign Zero, I developed the data merging strategy and built the New York City ETL pipeline, which processed the project's largest dataset (53M+ records). I also created a universal GEOID converter that performs spatial joins with Census shapefiles to enrich call data with geographic identifiers, enabling neighborhood-level analysis across all cities in the pipeline.</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Results & Impact</h4>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Scale:</strong> Successfully harmonized 60M+ 911 call records from NYC, Detroit, and Seattle into unified schema</li>
  <li><strong>Performance:</strong> Reduced data processing time by 70% through optimized chunked processing and Parquet compression</li>
  <li><strong>Insights:</strong> Enabled cross-city pattern analysis revealing universal temporal trends (summer peaks, 8-10 PM surges across all cities)</li>
  <li><strong>Urban Profiles:</strong> Discovered city-specific patterns—NYC dominated by quality-of-life concerns (60% of calls), Detroit shows higher violent crime incidents, Seattle higher traffic/property issues</li>
  <li><strong>Policy Impact:</strong> Provided Campaign Zero with evidence-based data infrastructure showing "one-size-fits-all" public safety assumptions are insufficient</li>
  <li><strong>Scalability:</strong> Pipeline architecture ready to ingest data from majority of major U.S. cities with minimal modifications</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Key Findings</h4>
<p style="margin-bottom: 20px;">The standardized dataset revealed that while emergency demand is deeply local—shaped by specific urban characteristics—the temporal dynamics of policing are remarkably consistent. All cities exhibit identical summer peaks and late-night activity surges, suggesting police demand is driven by universal human activity cycles rather than local policy.</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">What I Learned</h4>
<p style="margin-bottom: 0;">This project deepened my understanding of large-scale data engineering, distributed processing, and the importance of data standardization for meaningful analysis. I gained hands-on experience with geospatial data enrichment, ETL pipeline architecture, and collaborative development in a team environment. Working directly with Campaign Zero taught me how technical solutions can drive social impact through evidence-based policy research.</p>
</div>`,
    tech: ['Python', 'MS Azure', 'Pandas', 'GeoPandas', 'SQL', 'Jupyter', 'YAML', 'Parquet'],
    features: [
    'Configuration-driven ETL pipeline processing 60M+ records across 3 cities',
    'Chunked processing handling 53M NYC records in 105 batches (70% faster)',
    'Automated call type standardization mapping disparate terminologies to unified 15-category schema',
    'Geographic enrichment with Census TIGER/Line shapefiles for GEOID assignment',
    'Scalable architecture enabling new city ingestion with minimal code changes',
    'Comprehensive cross-city analysis revealing temporal patterns and urban safety profiles'],
    stats: [
      { value: '60M+', label: '911 Calls data' },
      { value: '11 Cities', label: '1 Schema' },
      { value: '15', label: 'Universal Categories' }
    ],
    github: 'https://github.com/BU-Spark/ds-cz-911/tree/fa25-team-a-dev',
    demo: null,
    report: 'files/Final Report_ 911 Call Data Patterns.pdf',
    hasGithub: true,
    hasDemo: false,
    hasReport: true 
  },
  'project3': {
    title: 'Stroke Risk Prediction: Multi-Model ML Analysis',
    category: 'Machine Learning',
    image: 'image/stroke.jpg',
    description: 'Project 3 description...',
    longDescription: `<div>
<h4 style="color: #00d4ff; margin-bottom: 12px; margin-top: 0;">Problem</h4>
<p style="margin-bottom: 16px;">Stroke is the 5th leading cause of death in the United States, but early prediction of stroke risk remains challenging due to complex interactions between patient symptoms, chronic conditions, and demographic factors. Traditional risk assessment methods often fail to capture the full spectrum of cardiovascular vulnerability or identify high-risk patients whose symptoms don't align with standard age-based expectations.</p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li>Need to identify patients with disproportionately high stroke risk for their age group</li>
  <li>Complex relationships between symptoms, chronic conditions, and overall health risk</li>
  <li>Challenge of detecting distinct patient risk profiles from high-dimensional symptom data</li>
  <li>Balance between model interpretability and predictive accuracy for clinical deployment</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Solution</h4>
<p style="margin-bottom: 20px;">Developed a comprehensive machine learning pipeline combining advanced feature engineering, unsupervised clustering for patient segmentation, and ensemble predictive models to classify stroke risk with 94% AUC. The solution integrates 7 engineered risk indices with K-Means and GMM clustering to identify distinct patient risk profiles, then applies Random Forest and XGBoost models for both regression (continuous risk prediction) and classification (binary risk assessment).</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Technical Implementation</h4>

<p style="margin-bottom: 12px;"><strong>Part 1: Feature Engineering (7 Risk Indices)</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>ANRI (Age-Normalized Risk Index):</strong> Identifies patients with unusually high stroke risk for their age by normalizing stroke risk percentage against age</li>
  <li><strong>CCS (Chronic Condition Score):</strong> Quantifies cardiovascular disease burden from high blood pressure and irregular heartbeat</li>
  <li><strong>SBI (Symptom Burden Index):</strong> Aggregates 8 symptom indicators (chest pain, shortness of breath, fatigue, dizziness, swelling, etc.) into total symptom load</li>
  <li><strong>AARZ (Age-Adjusted Risk Z-Scores):</strong> Compares individual risk to same-age peers using within-group standardization</li>
  <li><strong>RCI (Risk Consistency Index):</strong> Measures alignment between continuous risk scores and binary risk labels using pooled standard deviation</li>
  <li><strong>CHRI (Composite Health Risk Index):</strong> Weighted combination of chronic conditions and age-normalized risk (0.4 × BP + 0.4 × Heartbeat + 0.2 × ANRI)</li>
  <li><strong>Mutual Information Analysis:</strong> Ranked symptom predictive power using sklearn's mutual_info_classif to identify top contributors</li>
</ul>

<p style="margin-bottom: 12px;"><strong>Part 2: Unsupervised Clustering</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Feature Selection:</strong> StandardScaler normalization of age, SBI, CCS, ANRI, and stroke_risk_pct for clustering</li>
  <li><strong>K-Means Clustering:</strong> Elbow method and silhouette analysis to determine optimal k (tested k=2 to k=10), identified 4 distinct patient risk profiles</li>
  <li><strong>GMM (Gaussian Mixture Model):</strong> Probabilistic clustering allowing soft membership assignments, compared to K-Means using Adjusted Rand Index (ARI)</li>
  <li><strong>Cluster Profiling:</strong> Characterized each cluster by mean age, symptom burden, chronic conditions, and stroke risk percentage</li>
  <li><strong>Risk Stratification:</strong> Computed proportion of at-risk patients per cluster to validate clinical significance</li>
</ul>

<p style="margin-bottom: 12px;"><strong>Part 3: Predictive Modeling</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Regression Models:</strong> K-Nearest Neighbors, Random Forest, Linear Regression, XGBoost, and Gradient Boosting to predict continuous stroke_risk_pct. Evaluated using RMSE, MAE, and R²</li>
  <li><strong>Classification Models:</strong> Random Forest, KNN, Logistic Regression, XGBoost, SVM, and Naive Bayes to predict binary at-risk status. Evaluated using AUC, Accuracy, F1, and Balanced Accuracy</li>
  <li><strong>Feature Importance Analysis:</strong> Extracted and visualized top predictive features from best-performing models</li>
  <li><strong>Confusion Matrix Analysis:</strong> Detailed performance breakdown showing true/false positives and negatives</li>
  <li><strong>Pipeline Architecture:</strong> Scikit-learn pipelines with StandardScaler preprocessing for reproducible model training</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Results & Impact</h4>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Classification Performance:</strong> Achieved 94% AUC and 89% accuracy in predicting binary stroke risk using Random Forest with symptom-only features</li>
  <li><strong>Regression Performance:</strong> Random Forest achieved lowest RMSE (12.3) and highest R² (0.87) for continuous risk prediction</li>
  <li><strong>Risk Profile Discovery:</strong> K-Means clustering identified 4 distinct patient segments with stroke risk percentages ranging from 22% (low-risk young) to 78% (high-risk elderly with chronic conditions)</li>
  <li><strong>Feature Insights:</strong> Mutual Information analysis revealed chest pain, irregular heartbeat, and high BP as top 3 predictive symptoms, with MI scores 3x higher than baseline</li>
  <li><strong>High-Risk Identification:</strong> ANRI successfully flagged patients with ANRI > 2.9 (stroke risk 90-100% at age ~30), enabling early intervention</li>
  <li><strong>Clinical Validation:</strong> CCS analysis showed average stroke risk increases from 35% (0 conditions) to 68% (2 conditions), validating chronic disease impact</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Key Findings</h4>
<p style="margin-bottom: 16px;"><strong>Engineered Risk Indices:</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li>ANRI distribution revealed right-skewed tail with outliers at 2.9+, indicating subset of patients with disproportionate age-normalized risk</li>
  <li>SBI categorization (Low: 0-3, Moderate: 4-6, High: 7+) showed clear stratification with high-burden patients averaging 75% stroke risk vs 28% for low-burden</li>
  <li>CHRI composite index demonstrated stronger correlation with actual stroke outcomes than individual indicators alone</li>
</ul>

<p style="margin-bottom: 16px;"><strong>Patient Segmentation:</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li>Cluster 1: Young, healthy (age 42, risk 22%) - 15% at-risk</li>
  <li>Cluster 2: Middle-aged, moderate symptoms (age 58, risk 51%) - 48% at-risk</li>
  <li>Cluster 3: Elderly with chronic conditions (age 71, risk 78%) - 82% at-risk</li>
  <li>Cluster 4: Young with high symptom burden (age 35, risk 64%) - 67% at-risk (actionable outliers)</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">What I Learned</h4>
<p style="margin-bottom: 0;">This project strengthened my skills in advanced feature engineering, demonstrating how domain-specific risk indices can dramatically improve model interpretability and performance. I gained hands-on experience comparing supervised (Random Forest, XGBoost) and unsupervised (K-Means, GMM) approaches, understanding when probabilistic clustering adds value over hard assignments. The work reinforced the importance of comprehensive model evaluation using multiple metrics (AUC, F1, balanced accuracy) rather than relying on accuracy alone, especially for imbalanced medical datasets where false negatives carry high clinical costs.</p>
</div>`,
    tech: ['Python', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'XGBoost', 'Jupyter'],
    features: [
    '7 engineered risk indices (ANRI, CCS, SBI, AARZ, RCI, CHRI) for comprehensive health assessment',
    'K-Means and GMM clustering identifying 4 distinct patient risk profiles with 78% max risk variance',
    'Ensemble ML models (Random Forest, XGBoost, KNN) achieving 94% AUC for stroke classification',
    'Mutual Information feature ranking revealing top symptom predictors (chest pain, irregular heartbeat)',
    'Regression pipeline with RMSE 12.3 and R² 0.87 for continuous risk prediction',
    'Comprehensive model evaluation with confusion matrices, feature importance, and silhouette analysis'
  ],
    stats: [
      { value: '5', label: 'models' },
      { value: '94%', label: 'AUC Score' },
      { value: '89%', label: 'Accuracy' },
      { value: '7', label: 'Risk Indices' }
    ],
    github: 'https://github.com/dangdinh99/health_prediction_challenge',
    demo: null,
    report: null,
    hasGithub: true,
    hasDemo: false,
    hasReport: false
  },
  'project4': {
    title: 'Tree-or-Not: CNN Image Classifier',
    category: 'Machine Learning | Deep Learning ',
    image: 'image/treeornot.jpg',
    description: 'Project 4 description...',
    longDescription: `<div>
<h4 style="color: #00d4ff; margin-bottom: 12px; margin-top: 0;">Problem</h4>
<p style="margin-bottom: 16px;">Traditional CNN models on small datasets often suffer from overfitting and poor generalization. The baseline model achieved only 65.74% accuracy on validation data due to limited training images (292), high variance in composition (lighting, weather, indoor/outdoor), confounding objects (bushes, vegetation), and suboptimal hyperparameters.</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Solution</h4>
<p style="margin-bottom: 20px;">Implemented systematic ablation study testing 16 combinations of training improvements to identify optimal configuration. Through methodical experimentation with early stopping, dropout regularization, Xavier initialization, and learning rate scheduling, improved validation accuracy from 65.74% to 74.07% (+8.33 percentage points).</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Technical Implementation</h4>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Architecture:</strong> Custom CNN with convolutional layers, ReLU activation, max pooling, and fully connected classification head</li>
  <li><strong>Ablation Study:</strong> Systematically tested all 16 combinations of 4 improvements (Early Stopping, Xavier Init, LR Scheduler, Dropout) to isolate individual contributions</li>
  <li><strong>Winning Configuration:</strong> Early stopping (patience-based validation monitoring) + Dropout regularization (p=0.10)</li>
  <li><strong>Rejected Techniques:</strong> Xavier initialization and LR scheduler actually hurt performance—Xavier had neutral/negative effect, scheduler caused validation overfitting</li>
  <li><strong>GPU Pipeline:</strong> CUDA-accelerated training with efficient data preprocessing and checkpoint saving</li>
  <li><strong>Multiple Validation Sets:</strong> Used validation2 to test true generalization beyond initial validation split</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Results & Impact</h4>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Performance Gain:</strong> Baseline 65.74% → Improved 74.07% accuracy (+8.33 percentage points)</li>
  <li><strong>Efficiency:</strong> Simple regularization (dropout + early stopping) outperformed complex optimization schemes</li>
  <li><strong>Generalization:</strong> Model maintained performance across multiple validation sets, confirming robust generalization</li>
  <li><strong>Key Finding:</strong> Not all techniques help—Xavier init and LR scheduling degraded performance on small networks</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Key Insights</h4>
<p style="margin-bottom: 16px;">The ablation study revealed critical lessons about model optimization:</p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Less is More:</strong> Best models combined 2-3 complementary techniques rather than stacking all available improvements</li>
  <li><strong>Context Matters:</strong> Xavier initialization and LR scheduling—effective on large networks—actually hurt small network performance</li>
  <li><strong>Validation Strategy:</strong> LR scheduler overfit to validation set, only revealed through testing on validation2. Multiple validation sets essential for true generalization assessment</li>
  <li><strong>Regularization First:</strong> Simple dropout (10%) provided bigger gains than complex initialization or scheduling schemes</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">What I Learned</h4>
<p style="margin-bottom: 0;">This project taught me that systematic experimentation trumps intuition—not all "best practices" apply universally. I gained hands-on experience with ablation studies, understanding when techniques like Xavier initialization help (large networks) versus hurt (small networks). Building efficient GPU pipelines for image preprocessing and implementing model checkpointing with configuration tracking reinforced that engineering discipline is as important as modeling choices. The experience of discovering that LR schedulers can overfit to validation sets emphasized the importance of multiple validation strategies for assessing true generalization.</p>
</div>`,
    tech: ['Python', 'PyTorch', 'CUDA', 'NumPy', 'Matplotlib', 'Pandas', 'torcheval'],
    features: [
    'Systematic ablation study testing 16 hyperparameter combinations for optimization',
    'Custom CNN architecture with configurable dropout and early stopping',
    'GPU-accelerated training pipeline with CUDA optimization',
    'Multiple validation set strategy detecting scheduler overfitting to single validation split',
    'Model checkpointing with configuration tracking for reproducibility',
    'Efficient image preprocessing pipeline handling 256x256 RGB inputs'
    ],
    stats: [
      { value: '74%', label: 'Accuracy' },
      { value: '+8.3%', label: 'Improvement' },
      { value: '16', label: 'Configs Tested' }
    ],
    github: 'https://github.com/dangdinh99/tree-or-not',
    demo: null,
    hasGithub: true,
    hasDemo: false
  },
  'project5': {
    title: 'AI Agent Capabilities Evaluation Framework',
    category: 'AI Ethic | AI Bengmarking',
    image: 'image/ai_ethics.jpg',
    description: 'Rigorous evaluation framework benchmarking frontier AI models (Gemini 2.5, LLaMA 3.2) across 10 AI Ethics Index capability indicators aligned with NIST AI RMF.',
    longDescription:  `<div>
<h4 style="color: #00d4ff; margin-bottom: 12px; margin-top: 0;">Problem</h4>
<p style="margin-bottom: 20px;">As AI agents become increasingly deployed in production systems, there's a critical need for standardized evaluation of their operational reliability, safety, and interpretability. Existing benchmarks focus primarily on accuracy metrics, overlooking essential capabilities like error recovery, safety disclosures, educational alignment, state consistency, and explainability—factors that determine whether AI systems can be trusted in real-world applications. Without rigorous frameworks aligned with regulatory standards like NIST AI RMF and EU AI Act, organizations lack objective methods to assess model readiness for deployment.</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Solution</h4>
<p style="margin-bottom: 20px;">Developed a comprehensive evaluation framework assessing two frontier AI models (Gemini 2.5 Flash Lite and LLaMA 3.2-3B) across 10 critical capability indicators aligned with AI Ethics Index categories and NIST AI Risk Management Framework. The framework uses prompt-based behavioral testing across 100+ scenarios, automated scoring with expert validation, and independent ChatGPT-4 verification to provide objective, reproducible assessments of documentation completeness, safety policies, educational design, tool reliability, and interpretability.</p>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Technical Implementation</h4>

<p style="margin-bottom: 12px;"><strong>10 Evaluation Dimensions:</strong></p>
<ul style="margin-left: 20px; margin-bottom: 16px; line-height: 1.8;">
  <li><strong>L4.1 Intended Use Disclosures:</strong> Documentation of intended applications and explicit out-of-scope use cases</li>
  <li><strong>L4.2 Safety-Critical Restrictions:</strong> System policies explicitly disallowing high-risk deployment contexts</li>
  <li><strong>L4.3 Educational Objectives:</strong> Explicit learning goals, KPIs, baseline measurements, and evaluation plans</li>
  <li><strong>L4.4 Pedagogy Evidence:</strong> Instructional design principles, formative assessment, learning science alignment</li>
</ul>

<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>L4.5 Tool-Call Success & Error Recovery:</strong> Function execution accuracy and failure recovery in multi-step workflows</li>
  <li><strong>L4.6 Idempotence & Rollback:</strong> Handling duplicate requests and transaction reversibility</li>
  <li><strong>L4.7 State Consistency:</strong> Conversation context maintenance and reference accuracy</li>
  <li><strong>L4.8 Incident SLOs:</strong> Error handling, graceful degradation, and recovery time objectives</li>
  <li><strong>L4.9 Global Interpretability:</strong> Model documentation completeness and limitation transparency</li>
  <li><strong>L4.10 Local Explanations:</strong> Per-decision reasoning quality and transparency via API/UI</li>
</ul>

<p style="margin-bottom: 12px;"><strong>Methodology:</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Prompt Engineering:</strong> Designed 100+ scenario-based test cases covering documentation review (L4.1-L4.4) and behavioral testing (L4.5-L4.10)</li>
  <li><strong>Reproducibility:</strong> Temperature=0 for deterministic outputs, enabling consistent cross-model comparisons</li>
  <li><strong>Independent Validation:</strong> ChatGPT-4 validation achieving 70% exact agreement and 90% within-1 agreement on incident SLOs</li>
  <li><strong>Automated Pipeline:</strong> Self-contained Jupyter notebooks with integrated package installation, API integration, and automated scoring</li>
  <li><strong>Regulatory Alignment:</strong> Indicators mapped to NIST AI RMF, OECD AI Principles, and EU AI Act compliance requirements</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Key Findings</h4>
<p style="margin-bottom: 16px;"><strong>Documentation & Safety (L4.1-L4.4):</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li>Both models achieved 100% on global documentation completeness (L4.9), demonstrating strong transparency standards</li>
  <li>Safety-critical use case restrictions (L4.2) were explicit in documentation, aligning with emerging regulatory requirements</li>
  <li>Educational design evidence (L4.3-L4.4) varied between models, with commercial APIs providing more structured pedagogical frameworks</li>
</ul>

<p style="margin-bottom: 16px;"><strong>Operational Reliability (L4.5-L4.8):</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>LLaMA 3.2-3B:</strong> Superior operational capabilities with 70.4% tool-calling accuracy, 91.6% incident SLO compliance, and 87.5% state consistency—better suited for production systems requiring reliable function execution</li>
  <li><strong>Gemini 2.5 Flash Lite:</strong> Lower tool-calling reliability (51.9%) but stronger at conversational coherence, indicating design optimization for dialogue quality over operational precision</li>
  <li>Both models struggled with idempotence and rollback (L4.6), revealing gap in transaction-safe operation design</li>
</ul>

<p style="margin-bottom: 16px;"><strong>Interpretability (L4.9-L4.10):</strong></p>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Gemini 2.5 Flash Lite:</strong> 95.6% local explanation quality with nuanced reasoning, better choice for human-oversight applications</li>
  <li><strong>LLaMA 3.2-3B:</strong> 88.9% explanation quality, balancing operational reliability with interpretability</li>
  <li>Trade-off identified: conversational fluency vs operational precision in frontier model design</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">Impact & Applications</h4>
<ul style="margin-left: 20px; margin-bottom: 20px; line-height: 1.8;">
  <li><strong>Model Selection Framework:</strong> Organizations can objectively choose models based on deployment context—reliability-critical systems (LLaMA) vs interpretability-critical applications (Gemini)</li>
  <li><strong>Regulatory Compliance:</strong> Framework provides audit trail aligned with NIST AI RMF and EU AI Act transparency/safety requirements</li>
  <li><strong>Reproducible Research:</strong> Self-contained Jupyter notebooks enable researchers to replicate findings or extend framework to new models</li>
  <li><strong>Industry Benchmarking:</strong> Establishes baseline metrics for evaluating frontier models beyond traditional accuracy scores</li>
  <li><strong>Educational Standards:</strong> L4.3-L4.4 indicators provide rubric for assessing AI systems designed for learning contexts</li>
</ul>

<h4 style="color: #00d4ff; margin-bottom: 12px;">What I Learned</h4>
<p style="margin-bottom: 0;">This project deepened my understanding of AI safety and ethics beyond theoretical frameworks, showing how to operationalize regulatory requirements into measurable indicators across documentation, safety, pedagogy, and operational dimensions. I gained experience designing evaluation methodologies that balance automation with expert validation, and learned that model performance encompasses far more than accuracy—reliability, consistency, safety disclosures, and interpretability are equally critical for responsible deployment. Working with both commercial (Gemini) and open-source (LLaMA) models highlighted different design philosophies and revealed fundamental trade-offs between conversational quality and operational precision. The experience reinforced that ethical AI development requires rigorous, transparent evaluation frameworks that can adapt as models, use cases, and regulations evolve.</p>
</div>`,
    tech: ['Python', 'Gemini API', 'LLaMA 3.2', 'HuggingFace', 'Pandas', 'PyTorch', 'Jupyter'],
    features: [
    '10 AI Ethics Index indicators covering documentation, safety, pedagogy, and operations',
    'Comparative benchmarking of Gemini 2.5 Flash Lite vs LLaMA 3.2-3B across over 100 prompts scenarios',
    'Independent ChatGPT-4 validation with 70-90% inter-rater agreement',
    'NIST AI RMF and EU AI Act compliance mapping for regulatory alignment',
    'Self-contained Jupyter notebooks with automated package installation',
    'Reproducible methodology with Temperature=0 for deterministic evaluation'
  ],
    stats: [
      { value: '100+', label: 'Prompts ' },
      { value: '90%', label: 'Validation Agreement' },
      { value: '10', label: 'Capability Indicators' }
    ],
    github: 'https://github.com/dangdinh99/AI_Ethic_Benchmarking',
    demo: null,
    report: 'files/DS680_Assignment3_Bun_Bo.pdf',
    hasGithub: true,
    hasDemo: false,
    hasReport: true
  }
};

// Create modal HTML
function createModalHTML() {
  const modalHTML = `
    <div class="modal-overlay" id="projectModal">
      <div class="modal-content">
        <button class="modal-close" id="modalClose">
          <i class="fas fa-times"></i>
        </button>
        
        <div class="modal-header">
          <img src="" alt="" class="modal-header-image" id="modalImage">
          <div class="modal-header-overlay">
            <span class="modal-category" id="modalCategory"></span>
            <h2 class="modal-title" id="modalTitle"></h2>
          </div>
        </div>
        
        <div class="modal-body">
          <div class="modal-section">
            <h3><i class="fas fa-info-circle"></i> Overview</h3>
            <p class="modal-description" id="modalDescription"></p>
          </div>
          
          <div class="modal-section">
            <h3><i class="fas fa-tools"></i> Technologies Used</h3>
            <div class="modal-tech-grid" id="modalTech"></div>
          </div>
          
          <div class="modal-section">
            <h3><i class="fas fa-star"></i> Key Features & Achievements</h3>
            <ul class="modal-features-list" id="modalFeatures"></ul>
          </div>
          
          <div class="modal-section" id="modalStatsSection">
            <h3><i class="fas fa-chart-line"></i> Impact & Results</h3>
            <div class="modal-stats" id="modalStats"></div>
          </div>
        </div>
        
        <div class="modal-actions" id="modalActions"></div>
      </div>
    </div>
  `;
  
  document.body.insertAdjacentHTML('beforeend', modalHTML);
}

// Open modal with project data
function openProjectModal(projectId) {
  const project = projectsData[projectId];
  if (!project) return;
  
  const modal = document.getElementById('projectModal');
  
  // Populate modal content
  document.getElementById('modalImage').src = project.image;
  document.getElementById('modalImage').alt = project.title;
  document.getElementById('modalCategory').textContent = project.category;
  document.getElementById('modalTitle').textContent = project.title;
  document.getElementById('modalDescription').innerHTML = project.longDescription;
  
  // Populate tech stack
  const techContainer = document.getElementById('modalTech');
  techContainer.innerHTML = project.tech.map(tech => 
    `<span class="modal-tech-badge">${tech}</span>`
  ).join('');
  
  // Populate features
  const featuresContainer = document.getElementById('modalFeatures');
  featuresContainer.innerHTML = project.features.map(feature => 
    `<li><i class="fas fa-check-circle"></i> ${feature}</li>`
  ).join('');
  
  // Populate stats (if available)
  if (project.stats && project.stats.length > 0) {
    const statsContainer = document.getElementById('modalStats');
    statsContainer.innerHTML = project.stats.map(stat => 
      `<div class="modal-stat">
        <div class="modal-stat-value">${stat.value}</div>
        <div class="modal-stat-label">${stat.label}</div>
      </div>`
    ).join('');
  } else {
    document.getElementById('modalStatsSection').style.display = 'none';
  }
  
  // Populate action buttons
  const actionsContainer = document.getElementById('modalActions');
  let actionsHTML = '';

  if (project.hasGithub && project.github) {
    actionsHTML += `
      <a href="${project.github}" target="_blank" class="modal-btn modal-btn-primary">
        <i class="fab fa-github"></i> View on GitHub
      </a>
    `;
  }

  if (project.hasDemo && project.demo) {
    actionsHTML += `
      <a href="${project.demo}" target="_blank" class="modal-btn modal-btn-secondary">
        <i class="fas fa-external-link-alt"></i> Live Demo
      </a>
    `;
  }

  if (project.hasReport && project.report) {
    actionsHTML += `
      <a href="${project.report}" target="_blank" class="modal-btn modal-btn-secondary">
        <i class="fas fa-file-pdf"></i> View Report
      </a>
    `;
  }
  
  // If no links, add a close button
  if (!actionsHTML) {
    actionsHTML = `
      <button class="modal-btn modal-btn-primary" onclick="closeProjectModal()">
        <i class="fas fa-times"></i> Close
      </button>
    `;
  }
  
  actionsContainer.innerHTML = actionsHTML;
  
  // Show modal
  modal.classList.add('active');
  document.body.classList.add('modal-open');
}

// Close modal
function closeProjectModal() {
  const modal = document.getElementById('projectModal');
  modal.classList.remove('active');
  document.body.classList.remove('modal-open');
}

// Initialize modal system
document.addEventListener('DOMContentLoaded', () => {
  // Create modal
  createModalHTML();
  
  // Close button click
  document.getElementById('modalClose').addEventListener('click', closeProjectModal);
  
  // Click outside modal to close
  document.getElementById('projectModal').addEventListener('click', (e) => {
    if (e.target.id === 'projectModal') {
      closeProjectModal();
    }
  });
  
  // ESC key to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeProjectModal();
      
      // Also close mobile menu if open
      if (navMenu.classList.contains('show')) {
        navMenu.classList.remove('show');
        document.body.classList.remove('menu-open');
        menuToggle.classList.remove('fa-xmark');
        menuToggle.classList.add('fa-bars');
      }
    }
  });
  
  // Attach click handlers to "Learn More" buttons
  document.querySelectorAll('[data-project]').forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const projectId = button.getAttribute('data-project');
      openProjectModal(projectId);
    });
  });
  
  // Initialize scroll reveal animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in-up');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  const animateElements = document.querySelectorAll(
    '.project-card, .timeline-item, .cert-card, .contact-item, .skill-category'
  );
  
  animateElements.forEach(el => observer.observe(el));
  
  // Initialize active nav link
  updateActiveNavLink();
});

// === Performance: Debounce Scroll Events ===
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

const debouncedScroll = debounce(() => {
  updateActiveNavLink();
}, 10);

window.addEventListener('scroll', debouncedScroll);

// === Log for debugging ===
console.log('Portfolio with modal system loaded successfully! 🚀');
console.log('Projects available:', Object.keys(projectsData));