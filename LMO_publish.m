%% Modeling isotopic evolution of the lunar magma ocean
% Kelsey Prissel

% Crystallizes lunar magma ocean (LMO) according to different models in literature
% Input is LMO-PCS-modes.xlsx which contains PCS and phase comps for each model
% Models isotopic fractionation for 11 elements and 10 phases given user input
% mineral-melt fractionations. Current version does not contain temperature
% or oxygen fugacity controls on fractionation.

%%% Version history %%%
% Created 12/20/2015: User-defined D values and mineral stoichiometry
% Model results presented in Williams et al. (2016) LPSC 47, #2779

% Updated 11/11/2017: Modified for .xls input, matrices changed to cells

% Updated 11/28/2017: Uses experiment-defined partitioning for major elements
% in each mineral phase for experimental models

% Updated 2/28/2019: Only uses experimental models (Removed user-defined D
% value method, and thus, Snyder et al. 1992, Elkins-Tanton et al. 2011
% models removed). Now includes Rapp & Draper (2018), Charlier et al. (2018),
% and Lin et al. models. 
% Model results presented in Prissel & Krawczynski (2019) LPSC 50, #1912

% Updated 3/28/2020: Finalizing inputs for results reported in dissertation
% Model results presented in Prissel (2020), https://openscholarship.wustl.edu/art_sci_etds/2337/

% Updating 9/30/2022: Cleaning up and simplifying for public use. Added
% Schmidt and Kraettli (2022) crystallization sequence (Fe2TiO4)
% Updating 6/21/2023: Addressing review comments for publication

% Updating 10/19/2023: Adjusting ilmenite to use a fixed mineral major element composition as
% defined by the experiments


close all
clearvars
clc

%% Input file and define order for minerals and oxides
filename = 'LMO-PCS-modes-ilmenite.xlsx'; %input file

%%% Mineral order
% OL = 1, OPX = 2, CPX = 3, PLAG = 4, SP = 5, QTZ = 6, ILM = 7
% GARNET = 8, PIGEONITE = 9, APATITE = 10
mineral = {'ol','opx','cpx','plag','sp','qtz','ilm','gt','pig','ap'};

%%% Oxides order:
oxides = {'SiO2','TiO2','Al2O3','Cr2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','P2O5'};
ox = length(oxides);

%%% Initializing variables that will be defined by user in next sections
alpha{1,10} = []; %pre-allocate for speed
for a = 1:ox %Initialize mineral-melt fractionation factor for each element
    alpha{a} = ones(1,length(mineral)); %set alpha for each phase to 1 (no fractionation)
end

delta_liq0 = zeros(1,ox); %initialize delta isotopic compositions of liquid

%% Define fractionation factors (mineral-melt)
% Set fractionations for given elements/minerals in the format
% alpha{element index}(mineral index) = [];

%Fe isotopes (56/54)
% alpha{5} = [1 1 1 1 1 1 1 1 1 1]; % no fractionation...
% alpha{5} = [0.999860	0.999886	0.999913	1	1	1	1.000200	1	0.999913	1]; %Sedaghatpour & Jacobsen (2019)
% alpha{5} = [1	0.999982	1    1	1.000006	1	0.999945	1	1	1];	%ilm-2, NRIXS @ 1000°C, green glass
alpha{5} = [1	0.999982	0.999913    1	1.000006	1	0.999945	1	0.999913	1];	%NRIXS @1000°C but use SJ19 for cpx
%At high T, these olivine and opx fractionations go to 0.01‰
% alpha{5}(5) = [1.000006]; %NRIXS spinel fractionation

%Prissel et al. values:
% alpha{5}(7) = [0.999945]; %ilm-2, NRIXS @ 1000°C, green glass
% alpha{5}(7) = [1.000009];	%F144, NRIXS @ 1000°C, green glass

alpha{5}(7) = [0.99994]; %changing ilmenite value to Prissel (2020) - mineral melt 
% alpha{5}(7) = [1.00006]; %changing ilmenite value to Prissel (2020) + mineral melt


%Ti isotopes
% alpha{2} = [1	1	1.000060	1	1	1	0.999930	1	1.000060	1]; %IW+1 Rzehak et al. (2022)
alpha{2} = [1	1	0.999970	1	1	1	0.999540	1	0.999970	1]; %IW-1 Rzehak et al. (2022)
% alpha{2}(7) = 0.99991; % ilmenite-melt fxn, Prissel (2020) IW+1
% alpha{2}(7) = 0.999540; %ilmenite-melt fxn, Rzehak et al. (2022) IW-1


%Mg, isotopes
% alpha{7} = [1 1 1 1 1 1 1 1 1 1]; % no fractionation...
% alpha{7} = alpha{5}; %Mg is the same as Fe
% alpha{7} = [0.99998 0.99990 0.9998 1 1 1 0.9997 1 0.99990 1];% Mg estimates (some description in Chen et al. (2018))
alpha{7} = [1	1.000050	1.000155	1.000869	1	1	1	1	1.000155	1]; %Mg Sedaghatpour & Jacobsen (2019)
% alpha{7} = [1	1.000046	1.000124	1	1.000258	1	0.999859	1	1.000124	1];% Mg (Wang et al. 2023 force constants, assuming forsterite=melt at 1000°C)
% alpha{7}(7) = 0.99986; % ilmenite-forsterite, Wang et al. (2023) for most Fe rich geikielite-ilmenite given in Table 3

%Cr isotopes
alpha{4} = [1 1.00005 1.00005 1 1.00015 1 1 1 1.00005 1]; %Cr spinel-melt fxn Bonnand et al. (2016) %they do range but this is most consistent with Moynier et al (2011) according to their text

%Ca isotopes
alpha{8} = [1.000616	1.000452	1.000056	0.999963	1	1	1	1	1.000353	1]; %Huang et al. (2019), Klaver et al. (2021)
% alpha{8} = [1.000920	1.000520	1.000265	0.999850	1.	1.	1.	1.	1.000440	1.]; %Fu et al. (2023)



%% Define initial isotopic composition (Bulk Silicate Moon)
% Set initial liquid isotopic composition (delta) in the format
% delta_liq0(element index) = [];

delta_liq0(5) = 0.0627; %initial Fe 56/54 comp of liquid
delta_liq0(2) = -0.003; %initial Ti 49 comp of liquid
delta_liq0(7) = -0.26; %initial Mg 26/24 comp of liquid
delta_liq0(4) = -0.222; %initial Cr 53/52 comp of liquid
delta_liq0(8) = 0.879; %initial Ca 44/40 comp of liquid (Fu et al. 2023)


%% Fractionation code
dF = 0.01; %amount of remaining liquid crystallized at each step (0.01 = 1%)
xln = 10/dF; %number of fractionation steps in crystallization

liqpct(1) = 1; %initial liquid fraction (0-1)

for ii = 2:xln
    liqpct(ii) = liqpct(ii-1) - (liqpct(ii-1)*dF); %amount of liquid left, fraction between 0 and 1
end

% PCS is calculated as a function of remaining liquid amount at each step
PCS_calc = 1-liqpct; %PCS fraction = 1 - liquid fraction
PCS_calc = PCS_calc*100; %turn PCS fraction into percent

[type,sheetname] = xlsfinfo(filename); %load input .xls file
model_tabs = size(sheetname,2); %number of tabs (models) in spreadsheet

models = 1:model_tabs; %tab indices for models to use in calculation
% models = 4;
% Use 1:model_tabs to run all models in the spreadsheet

%%% Notes on indexing:
% m is for # models & indexing, in order of spreadsheet
% z is for # of phases (l also for indexing phases)

for t = 1:length(models) %for every model chosen above
    m = models(t); %set m equal to model tab index
    name{m} = char(sheetname(1,m)); %get name of model from tab m
    [modes{m,1}, txt{m}] = xlsread(filename, m); %data matrix and text from tab m
    
    steps = size(modes{m},1); % #rows in data matrix (PCS steps)
    columns = size(modes{m},2); % #cols in data matrix (# phases + 1)
    headers{m} = txt{m}(1,:); %column headers from each tab (PCS, phase names)
    
    % Take comma-delimited input compositions for each phase at each PCS
    % Liquid is in col 1, liquid fraction is 100-PCS (PCS is data column 1)
    comp_comma = txt{m}(steps+2:end,:); %all "text" composition inputs, first row (3) to last row
    
    % Convert each column array of comma-delimited strings to a matrix
    % Syntax taken from https://www.mathworks.com/matlabcentral/answers/345742-convert-array-of-comma-separated-strings-in-cells-to-matrix
    for p = 1:columns
        Arr = comp_comma(:,p);
        Str = sprintf('%s,', Arr{:});
        phase_comps{m}{p} = sscanf(Str, '%g,', [ox,inf]).';
    end
    
    PCS{m} = modes{m}(:,1); %PCS column from tab m
    liq{m} = phase_comps{m}{1}; %liquid compositions, liquid is 1st column
    bulk{m} = liq{m}(1,:); %bulk composition is liq at 0 PCS (first entry)
    
    for c = 1:length(delta_liq0) %for each element
        delta_liq{m}{c}(1) = delta_liq0(c); %set initial isotopic comp of liquid
    end
    
    for z = 1:columns-1 %for each phase
        D{m}{z} = phase_comps{m}{z+1}./liq{m}; %define mineral-liquid distribution for major element oxides
    end
    
    Kd_ol{m} = D{m}{1}(:,5)./D{m}{1}(:,7); %Kd, Fe/Mg for ol, ol is 1st phase (z=1)
    
    % Calculate mole fractions of bulk composition
    % [SiO2, TiO2, Al2O3, Cr2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5];
    molwt = [60.08, 79.9, 101.96, 152, 71.85, 70.94, 40.30,...
        56.08, 61.98, 94.20 283.886]; %molar weight, oxide g/mol
    oxmol = bulk{m}./molwt; %oxide moles
    summol = sum(oxmol); %sum of oxide moles for normalization
    
    liquid{m}(1,:) = oxmol./summol; %bulk liquid composition in mol fraction
    bulk_solid{m} = zeros(length(liqpct),size(phase_comps{m}{1},2),1); %initialize bulk solid composition 
    
    % Convert wt% minerals to mol% minerals
    sumphasemol{m} = zeros(size(phase_comps{m}{1},1),1); %preallocate
    for p = 2:length(phase_comps{m}) %for each solid phase
        totalmol{m}{p} = phase_comps{m}{p}*molwt'; %multiplies each element by molwt of that element then sums mols for mineral
        molmode{m}{p} = modes{m}(:,p).*totalmol{m}{p}; %multiples mass pct times molar weight of mineral
        sumphasemol{m} = sumphasemol{m}+molmode{m}{p}; %sums for each mineral at a given step
    end
    for p = 2:length(phase_comps{m})
        molfrac{m}(:,p) = molmode{m}{p}./sumphasemol{m}; %divides each mineral mol*mass by the sum to get mol frac of mineral
    end
    
    % This nested loop steps through calculation PCS array (PCS_calc) and
    % assigns modal mineralogy and D values as defined by each model.
    % D values for each oxide in each mineral will be used within the 
    % experimental PCS interval that defined D and modes

    for s = 1:length(liqpct) %for each fractionation step
        for k = steps:-1:1 %Start by assigning through to last step, then step backward and "rewrite" earlier ones
            if PCS_calc(s) <= PCS{m}(k) %check whether current PCS_calc value (s) is before PCS interval boundary (k) in given model {m}
                modes_full{m}(s,:) = molfrac{m}(k, 2:columns); %write modal mineralogy to PCS_calc step for given PCS interval (k)
                for f = 1:z %then, for every phase
                    %Assign D for all major element oxides in each mineral to current PCS_calc step
                    D_full{m}{f}(s,:) = D{m}{f}(k,:); %get major element D's for phase(f)-liq
                    phase_comp_full{m}{f}(s,:) = phase_comps{m}{f+1}(k,:); %tracking major element phase comps to use in fractionation calculation
                end
            end
        end
    end
    

    for i = 1:length(liqpct) %going through the incremental PCS steps
        mol2wt = liquid{m}(i,:).*molwt; %making wt% matrix at each step
        liq_wt{m}(i,:) = mol2wt./sum(mol2wt)*100; %normalize wt% to 100
        
        %{l} values OL = 1, OPX = 2, CPX = 3, PLAG = 4, SP = 5, QTZ = 6, ILM = 7, GARNET = 8, PIGEONITE = 9, APATITE = 10
        %Crystallization loop using the partitioning values from
        %experimental models
        
        phases = length(D_full{m}); %number of phase columns for model m

        for l = 1:phases %indexing which phase for loop
            if l == 4 && PCS_calc(i) > 98 %Experiment defined D method doesn't work as well toward final steps as liquid gets extreme compositions
                D_full{m}{l}(i,:) = phase_comp_full{m}{l}(i,:)./liq_wt{m}(i,:); %use reported mineral comp to define pseudo-D at this step
            elseif l == 7 %for every ilmenite, fix the mineral composition to be similar to that reported
                D_full{m}{l}(i,:) = phase_comp_full{m}{l}(i,:)./liq_wt{m}(i,:); %use reported mineral comp to define pseudo-D at this step
            end

            for d = 1:ox %check each element for NaN or Inf values because they mess up calculation
                if isnan(D_full{m}{l}(i,d)) || isinf(D_full{m}{l}(i,d)) %if there are NaN or Inf values
                    D_full{m}{l}(i,d) = 0; %set to 0 so mineral will not take any of that element
                end
            end

            comps{m}{l}(i,:) = D_full{m}{l}(i,:).*liquid{m}(i,:); %mol composition of mineral
            min_wt{m}{l}(i,:) = 100*(comps{m}{l}(i,:).*molwt)./sum(comps{m}{l}(i,:).*molwt); %wt% composition mineral

            if any(isnan(comps{m}{l}(i,:))) %if any NaN in composition
                comps{m}{l}(i,:) = NaN; %make all NaN
            end
            solid_prop{m}{l}(i,:) = comps{m}{l}(i,:).*modes_full{m}(i,l); %multiply solid composition by solid proportion
            bulk_solid{m}(i,:) = bulk_solid{m}(i,:) + solid_prop{m}{l}(i,:); %sum proportioned solid compositions of all phases (l)
        end

        Cl = liquid{m}(i,:); %mol comp of liquid
        Cs = bulk_solid{m}(i,:); %mol comp of bulk solid
        
        sol2wt = Cs.*molwt; %making wt% matrix for solid
        sol_wt{m}(i,:) = sol2wt./sum(sol2wt)*100; %normalize wt% to 100%
        
        %For looking at cumulate with 100% removal of plagioclase
        %(efficient segregation for plagioclase crust formation)
        sol2wt_noplag = (Cs-solid_prop{m}{4}(i,:)).*molwt; %take out plag (phase 4)
        sol_wt_noplag{m}(i,:) = sol2wt_noplag./sum(sol2wt_noplag)*100; %normalize wt% to 100%
        
        if any(isnan(Cs)) 
            Cs = zeros(1,length(Cs)); %eliminate error from first and last steps having "no" solid = NaN values
        end
        
        %%% BULK FRACTIONATION CALCULATION
        bulkD{m}(i,:) = Cs./Cl; %calculate the bulk D for each element
        liquid{m}(i+1,:) = Cl - dF*Cs; %calculate new liquid composition unnormalized
        
        LIQFRAC{m}(i,:) = liquid{m}(i,:)./liquid{m}(1,:); %ratio of remaining liquid mol to intial liquid mol
        solfrac{m}(i,:) = ones(1,length(LIQFRAC{m}(i,:))) - LIQFRAC{m}(i,:); %molfrac of each element in the solid (summative)
        
        if i >1 %if crystallization has begun
            solstep{m}(i,:) = solfrac{m}(i,:) - solfrac{m}(i-1,:); %mol fraction of element out by solid at each iteration
        else solstep{m}(i,:) = zeros(1,length(LIQFRAC{m}(i,:))); %no solid crystallizes at initial step
        end

        %%% Check for negative values in new liquid
        if any(liquid{m}(i+1,:)<0) %check if ANY element is negative
            index = find(liquid{m}(i+1,:)<0); %find which element is negative
            liquid{m}(i+1, index) = 0; %remove the negative element
            
            disp(['Model liquid ' num2str(m) ' ran out of ' oxides{index}...
                ' at PCS = ' num2str(round(PCS_calc(i),2))]); %tell user what element has reached negative, in which model, and at which PCS

            if any(index == [1 2 3 5 7 8])%if a major element has run out
                liquid{m}(i+1, :) = NaN; %remove the negative element; NaN will make code "stop" (all NaN) vs. "0" above code continues just without that element
            end

        end

        for f = 1:ox %for each element
            for l = 1:phases %for each phase
                delta_sol{m}{f}(i,l) = alpha{f}(l)*(delta_liq{m}{f}(i) + 1000)-1000; %calculate instantaneous delta composition for each mineral
                elem_frac{m}{f}(i,l) = solid_prop{m}{l}(i,f)./Cs(f); %fraction of the element out in each mineral
            end
            
            if isnan(elem_frac{m}{f}(i,:)) %eliminate error from first step having no solid
                elem_frac{m}{f}(i,:) = 0;
            end
            
            delta_bulk_sol{m}{f}(i) = delta_sol{m}{f}(i,:)*elem_frac{m}{f}(i,:)'; %instantaneous bulk solid delta composition
            delta_no_plag{m}{f}(i) = delta_bulk_sol{m}{f}(i) - delta_sol{m}{f}(i,4).*elem_frac{m}{f}(i,4); %instantaneous bulk solid delta composition without plag
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%% CALCULATION FOR LIQUID ISOTOPIC EVOLUTION %%%%
            
            instsol{m}{f}(i) = solstep{m}(i,f).*delta_bulk_sol{m}{f}(i); %weight instantaneous bulk solid delta composition by the element frac taken out of liquid by solid
            sumsol{m}{f}(i) = sum(instsol{m}{f}(1:i)); %sum the weighted deltas to get bulk solid delta up to this point
            %this works because we are stepping in equal PCS intervals, if
            %they were uneven, we would need to factor that in here

            delta_liq{m}{f}(i+1) = (delta_liq0(f) - sumsol{m}{f}(i))./LIQFRAC{m}(i,f);
            %derived from: bulk liquid = liqfrac*liq comp + solfrac*sol comp
            %liquid delta composition at this step is equal to the initial liquid delta composition
            %minus the bulk solid delta composition up to this point weighted by liquid fraction remaining from initial

%%% Removing zeros and turning to NaN for plotting purposes
            for h = 1:l
                if solid_prop{m}{h}(i,:) == zeros(1,length(oxides))
                    delta_sol{m}{f}(i,h) = NaN; %doing same thing that was done in Line 196-198
                    % to comps matrices if a mineral is not present
                    % Removing now because NaN would mess up bulk solid isotope
                    % compositon (zeros will not affect calculation because fraction is 0,
                    % but zeros will show up in plots, so prefer to have NaN)
                end
                if elem_frac{m}{f}(i,h) == 0 %If mineral doesn't have that element
                    delta_sol{m}{f}(i,h) = NaN; %No delta composition for element
                end
            end
        end
    end
    
    for v = 1:length(liqpct)
        if isnan(bulk_solid{m}(v,:)) %if mineral is no longer crystallizing
            bulk_solid{m}(v,:) = NaN; %remove (for plotting)
            liq_wt{m}(v,:) = NaN; %doing this after next liquid has been defined so OK!
            for f = 1:length(oxides)
                delta_liq{m}{f}(v) = NaN;
                delta_bulk_sol{m}{f}(v) = NaN;
            end
        end
    end
    
    for c = 1:size(liq{m},1)
        if liq{m}(c,:) == zeros(1,length(oxides))
            liq{m}(c,:) = NaN; %remove 0's at PCS = 100 liquid comps (for plotting)
        end
    end
    
end

%% PLOTTING MODEL RESULTS
%% LMO liquid major oxide compositions
% figure(1)
% for t = 1:length(models)
%     b = models(t);
%     %oxides = {'SiO2','TiO2','Al2O3','Cr2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','P2O5'};
%     oxplot = [1 2 3 5 7 8];
%     %oxplot = 1:11; %to plot all oxides
%     for j = 1:length(oxplot)
%         subplot(3,3,j)
%         hold on
%         plot(PCS_calc, liq_wt{b}(:,oxplot(j)),'k') %model values
%         scatter(PCS{b}, liq{b}(:,oxplot(j)), 'r') %data from experiments
%         %ylim([0 100])
%         %xlim([80 100])
%         xlabel('Percent LMO solidified')
%         ylabel(['wt% ' oxides{oxplot(j)}])
%         title(oxides{oxplot(j)})
%         hold off
%         box on
% 
%         subplot(3,3,7:9)
%         hold on
%         box on
%         plot(PCS_calc, liq_wt{b}(:,7)./(liq_wt{b}(:,7)+liq_wt{b}(:,5)).*100, 'k')
%         scatter(PCS{b}, liq{b}(:,7)./(liq{b}(:,7)+liq{b}(:,5)).*100, 'r')
%         ylim([0 100])
%         xlabel('Percent LMO solidified')
%         ylabel('Mg#')
%         hold off
%     end
% end
% hold off

%%
%to customize color of each phase in plots
color_set= [.165 .384 .275
    .467 .674 .188
    1 .6 .784
    .729 .831 .957
    1 1 0
    .494 .184 .557
    1 0 0
    .635 .078 .184
    .078 .169 .549
    0 1 1];
% 
% figure(4) %Plot Fe isotope comps for phases at each PCS
% for t = 1:length(models) %for each model
%     b = models(t); %get index for each model
%     subplot(4,5,t)
%     hold on
%     plot(PCS_calc, delta_liq{b}{5}(1:end-1)) %liquid composition @ each PCS
%     plot(PCS_calc, delta_bulk_sol{b}{5}) %bulk solid composition @ each PCS
%     for g=1:h
%         plot(PCS_calc, delta_sol{b}{5}(:,g),'Color',color_set(g,:)) %plot every solid's delta
%     end
% end
% hold off
% 
%average comps and ranges for each element (row = element #)
%low Ti, range, high Ti, range, KREEP, range
mare = [
    0 0 0 0 0 0 %SiO2
    -.003 .014 0.02 0.02 0.33 0.034 %49Ti basalts Millet, KREEP, Greber et al 2017
    0 0 0 0 0 0 %Al2O3
    -.215 .058 -.229 .074 -.158 .022 %53Cr Bonnand et al. 2016
    0.05 0.05 0.14 0.04 0.25 0 %56Fe
    0 0 0 0 0 0 %MnO
    -.285 .109 -.462 .084 -.349 0.038 %26Mg %sed. & jacob. 2018
    0 0 0 0 0 0 %CaO
    0 0 0 0 0 0 %Na2O
    -.07 0.09 -0.07 0.09 0.25 0 %K2O %from talking to Zhen, March 2019 %FHT -2.47 to -0.12 (-0.4 upper crust)
    0 0 0 0 0 0 %P2O5
    ];

%% Plot isotope evolution for element(s) of choice
figure(6)
%Define element indices (Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P)

choose = [5 2 7 4 8]; %Fe, Ti, Mg, Cr, Ca
n=length(choose);
for w = 1:n
    subplot(n,1,w)
    hold on
    k = choose(w); %element index that will plot this loop iteration
    for b = 1:length(models)
        for g=1:l
            plot(PCS_calc, delta_sol{models(b)}{k}(:,g),'Color',color_set(g,:)) %plot every solid's delta
        end
        plot(PCS_calc, delta_liq{models(b)}{k}(1:end-1),'k')
        %plot(x, delta_bulk_sol{models(b)},'k')
        %xlim([80 100])
        %ylim([-0.5 0.5])
        box on
        xlabel('PCS')
        ylabel('\delta composition')
        title(oxides{k})

        lowTi = [mare(k,1) mare(k,2)];
        highTi = [mare(k,3) mare(k,4)];
        KREEP = [mare(k,5) mare(k,6)];

        plot(PCS_calc,ones(1,length(PCS_calc)).*lowTi(1),'--g')
        plot(PCS_calc,ones(1,length(PCS_calc)).*highTi(1),'--b')
        plot(PCS_calc,ones(1,length(PCS_calc)).*KREEP(1),'--m')

        xlim([0 100])

        if k == 7 %Mg
            ylim([-1 1])
        elseif k == 2 %Ti axes
            ylim([-0.2 0.4])
            %elseif k == 4 %Cr
            %elseif k ==5 %Fe
            %xlim([86 100])
        end
    end
end
hold off

%%
color_set= [0 0.4 0.37
    0.21 0.59 0.56
    0.5 0.8 0.76
    0.96 0.91 0.76
    0.87 0.76 0.49
    0.55 0.32 0.04
    0.15 0.15 0.15
    1 1 1
    0.78 0.92 0.9
    1 1 1]; %colors that match the LMO stack figure


oxlabel = {'SiO_2','TiO_2','Al_2O_3','Cr_2O_3',...
    'FeO','MnO','MgO','CaO','Na_2O','K_2O', 'P_2O_5'};
oxplot = [5 2];
n=length(oxplot);

linesty = {'-', '-', '-', '-','-'}; %Can differentiate models by line type here

 %% Major element evolution plots for Fe and Ti
clf(figure(8))

for t = 1:length(models)
    %if t == 3
    %else
        b = models(t);
        %oxplot = 1:11; %to plot all oxides
        for j = 1:length(oxplot)
            %subplot(2,2,j)
            figure(8)
            subplot(1,2,j)
            hold on
            plot(PCS_calc, liq_wt{b}(:,oxplot(j)), 'k', 'LineWidth', 2) %model values
            plot(PCS{b}, liq{b}(:,oxplot(j)), 'o',...
                'MarkerSize', 8, 'LineWidth', 2, 'Color', [0.9 0 0]) %data from experiments
            
            xlabel('Percent LMO solidified')
            ylabel(['wt.% ' oxlabel{oxplot(j)}])
            box on
            set(gca, 'FontSize', 30, 'FontName', 'Myriad Pro','LineWidth', 2)
            legend('Model liquid', 'Experiment glass', 'Location', 'northwest')
            
            xlim([0 100])
            if oxplot(j) == 5 %Fe
                ylim([0 40])
            elseif oxplot(j) == 2 %Ti
                ylim([0 8])
            end
        end
    %end
end

 %% Major element evolution plots for Fe and Ti
clf(figure(11))
minidx = 7; %give index for mineral of interest
%OL = 1, OPX = 2, CPX = 3, PLAG = 4, SP = 5, QTZ = 6, ILM = 7, GARNET = 8, PIGEONITE = 9, APATITE = 10
oxplot = [5 2];

for t = 1:length(models)
    %if t == 3
    %else
        b = models(t);
        %oxplot = 1:11; %to plot all oxides
        for j = 1:length(oxplot)
            %subplot(2,2,j)
            figure(11)
            subplot(1,2,j)
            hold on
%             plot(PCS_calc, liq_wt{b}(:,oxplot(j)), 'k', 'LineWidth', 2) %model liquid values
%             plot(PCS{b}, liq{b}(:,oxplot(j)), 'o',...
%                 'MarkerSize', 8, 'LineWidth', 2, 'Color',color_set(7,:)) %glass data from experiments

            plot(PCS_calc, min_wt{b}{minidx}(:,oxplot(j)), 'k', 'LineWidth', 2) %model mineral values
            plot(PCS{b}, phase_comps{b}{minidx+1}(:,oxplot(j)), 'o',...
                'MarkerSize', 8, 'LineWidth', 2, 'Color',color_set(7,:)) %mineral data from experiments
            
            xlabel('Percent LMO solidified')
            ylabel(['wt.% ' oxlabel{oxplot(j)}])
            box on
            set(gca, 'FontSize', 30, 'FontName', 'Myriad Pro','LineWidth', 2)
            legend('Model mineral', 'Experiment mineral', 'Location', 'northwest')
            
            xlim([0 100])
            if oxplot(j) == 5 %Fe
%                 ylim([20 90])
            elseif oxplot(j) == 2 %Ti
%                 ylim([10 80])
            end
        end
    %end
end


%% Isotope evolution plots - zoom in on last 15%
clf(figure(9))
for t = 1:length(models)
    b = models(t);
    %oxplot = 1:11; %to plot all oxides
    for j = 1:length(oxplot)
        figure(9)
        %subplot(2,2,j+2)
        subplot(1,2,j)
        hold on
        k = oxplot(j); %element index that will plot this loop iteration
        for g=1:l
            pn{g} = plot(PCS_calc, delta_sol{b}{k}(:,g), linesty{t},...
                'Color',color_set(g,:), 'LineWidth', 3); %plot every solid's delta
        end
        
        pliq = plot(PCS_calc, delta_liq{b}{k}(1:end-1),[linesty{t} 'k'], 'LineWidth', 3, 'Color', [0.5 0.5 0.5]);
        
        xlim([85 100])
        ytickformat('%0.2f')

        xlabel('Percent LMO solidified')
        if oxplot(j) == 5
            ylabel(['\delta^{56}Fe (' char(8240) ')'])
            ylim([-0.05 0.35])
            ytickformat('%0.2f')
        elseif oxplot(j) == 2
            ylabel(['\delta^{49}Ti (' char(8240) ')'])
            ylim([-0.5 1.5])
            ytickformat('%0.1f')
        end
        
        set(gca, 'FontSize', 30, 'FontName', 'Myriad Pro', 'LineWidth', 2)
        box on
        
        %         lowTi = [mare(k,1) mare(k,2)];
        %         highTi = [mare(k,3) mare(k,4)];
        %         KREEP = [mare(k,5) mare(k,6)];
        %
        %         HT = plot(linspace(0,100,10),ones(1,10).*highTi(1),'--','LineWidth', 2, 'Color', color_set(7,:));
        %         KP = plot(linspace(0,100,10),ones(1,10).*KREEP(1),'--k','LineWidth', 2);
        
        if j == 1
            %LT = plot(linspace(0,100,10),ones(1,10).*lowTi(1),'--', 'LineWidth', 2,'Color', color_set(3,:));
%             lgd = legend([pn{7} HT pliq KP],...
%                 {'Ilmenite', 'High-Ti','Liquid', 'KREEP'}, 'Location', 'northoutside', 'FontSize', 30);
%             lgd.NumColumns = 3;
%             lgd = legend([pn{7} pliq],...
%                 {'Ilmenite', 'Liquid'}, 'Location', 'north', 'FontSize', 30);
%             lgd.NumColumns = 2;
        elseif j == 2
%             lgd = legend([pn{7} HT pliq  KP],...
%                 {'Ilmenite', 'High-Ti','Liquid',   'KREEP'}, 'Location', 'northoutside', 'FontSize', 30);
%             lgd.NumColumns = 3;
%             lgd = legend([pn{7} pliq],...
%                 {'Ilmenite', 'Liquid'}, 'Location', 'north', 'FontSize', 30);
%             lgd.NumColumns = 2;
        end
    end
end
hold off

%% Isotope evolution plots, full
clf(figure(10))
for t = 1:length(models)
    b = models(t);
    %oxplot = 1:11; %to plot all oxides
    for j = 1:length(oxplot)
        figure(10)
        %subplot(2,2,j+2)
        subplot(1,2,j)
        hold on
        k = oxplot(j); %element index that will plot this loop iteration
        for g=1:l
            pn{g} = plot(PCS_calc, delta_sol{b}{k}(:,g), linesty{t},...
                'Color',color_set(g,:), 'LineWidth', 3); %plot every solid's delta
        end
        
        pliq = plot(PCS_calc, delta_liq{b}{k}(1:end-1),[linesty{t} 'k'], 'LineWidth', 3, 'Color', [0.5 0.5 0.5]);
        
        xlim([0 100])
        
        xlabel('Percent LMO solidified')
        if oxplot(j) == 5
            ylabel(['\delta^{56}Fe (' char(8240) ')'])
            ylim([-0.05 0.35])
            ytickformat('%0.2f')

        elseif oxplot(j) == 2
            ylabel(['\delta^{49}Ti (' char(8240) ')'])
            ylim([-0.5 1.5])
            ytickformat('%0.1f')
        end
        
        set(gca, 'FontSize', 30, 'FontName', 'Myriad Pro', 'LineWidth', 2)
        box on
        
        %         lowTi = [mare(k,1) mare(k,2)];
        %         highTi = [mare(k,3) mare(k,4)];
        %         KREEP = [mare(k,5) mare(k,6)];
        %
        %         HT = plot(linspace(0,100,10),ones(1,10).*highTi(1),'--','LineWidth', 2, 'Color', color_set(7,:));
        %         KP = plot(linspace(0,100,10),ones(1,10).*KREEP(1),'--k','LineWidth', 2);
        
        if j == 1
            %LT = plot(linspace(0,100,10),ones(1,10).*lowTi(1),'--', 'LineWidth', 2,'Color', color_set(3,:));
%             lgd = legend([pn{7} HT pliq KP],...
%                 {'Ilmenite', 'High-Ti','Liquid', 'KREEP'}, 'Location', 'northoutside', 'FontSize', 30);
%             lgd.NumColumns = 3;
%             lgd = legend([pn{7} pliq],...
%                 {'Ilmenite', 'Liquid'}, 'Location', 'north', 'FontSize', 30);
%             lgd.NumColumns = 2;

        elseif j == 2
%             lgd = legend([pn{7} HT pliq  KP],...
%                 {'Ilmenite', 'High-Ti','Liquid',   'KREEP'}, 'Location', 'northoutside', 'FontSize', 30);
%             lgd.NumColumns = 3;
%             lgd = legend([pn{7} pliq],...
%                 {'Ilmenite', 'Liquid'}, 'Location', 'north', 'FontSize', 30);
%             lgd.NumColumns = 2;
        end
    end
end
hold off