
classdef TPA_EventManager %< handle
    % TPA_EventManagerClass - defines Last Event management class
    % Should change the future management of Events - Behavioral ROIs
    % Inputs:
    %       global variables and more
    % Outputs:
    %        different functions
    
    %-----------------------------
    % Ver	Date	 Who	Descr
    %-----------------------------
    % 23.08 22.03.16 UD     Adding class from Jaaba
    % 21.02 18.08.15 UD     merging with RoiManager
    % 20.04 17.05.15 UD     Created
    %-----------------------------
  
    properties (Constant)
        % global constants see Par file
        VERSION             = '2308'; % SW version supported
        
    end
    
    properties
        % Selected ROI properties
        % init common ROI structure
        Type                = 0;               % define ROI type
        State               = 0;               % designates in which init state ROI is in
        Color               = rand(1,3);       % generate colors
        Name                = 'X';             % which name
        SeqNum              = 0;               % event numbering
        %AverType = 0;                         % which operation to perform
        CellPart            = 1;
        CountId             = 1;               % TBD
        Class               = 0;                % 0- none
        Id                  = 0; 
        TimeInd             = [];
        
        xyInd               = [];          % shape in xy plane
        ytInd               = [];          % shape in xy plane
        
        % bounding box
        xInd                = [1 1];           % range location in X
        yInd                = [1 1];           % range location in Y
        zInd                = [1 1];           % range location in Z stack
        tInd                = [1 1];           % rangelocation in T stack
        
        % Data 
        Data                = [];              % processed data for ROI
        
        % init graphics
        ViewType            = 1;               % which type default
        NameShow            = false;           % manage show name
        ViewXY              = [];             % structure contains XY shape params
        ViewYT              = [];             % structure contains YT shape params
        
        % clicks position
        pointRef                    = [10 10];  % point1        
        rectangleInitialPosition    = [-1 -1 -1 -1]; %rectangle to move
        shapeInitialDrawing         = [0 0]; % coordinates of the shape
        
        % internal stuff
        cntxMenu            = [];  % handle that will contain context menu
        
        %Testing
        hFigure             = [];
        hAxes               = [];
        hImage              = [];
                    
        
        
    end
    
    properties (SetAccess = private)
%         % global constants see Par file
        STATE_TYPES         = struct('NONE',1,'INIT',2,'VIEWXY',3,'VIEWYT',4,'VIEWALL',5);
        VIEW_TYPES          = struct('XY',1,'YT',2,'XYYT',3);
        ROI_TYPES           = struct('RECT',1,'ELLIPSE',2,'FREEHAND',3);
        
    end
    
    methods
        
        % =============================================
        function obj = TPA_EventManager(roiType,viewType,hAxes)
            % Constructor

            % state of the ROI
            %obj.State               = obj.STATE_TYPES.NONE;
            % reinit
            obj.STATE_TYPES         = struct('NONE',1,'INIT',2,'VIEWXY',3,'VIEWYT',4,'VIEWALL',5);
            obj.VIEW_TYPES          = struct('XY',1,'YT',2,'XYYT',3);
            obj.ROI_TYPES           = struct('RECT',1,'ELLIPSE',2,'FREEHAND',3);

            
            % check if oter init is required
            % init object
            if nargin < 1, return; end;
            if nargin < 2, return; end;
            obj         = Init(obj,roiType,viewType);
            
            % init view
            if nargin < 3, return;   end;
            obj         = InitView(obj,viewType,hAxes);
                        
        end
        
%         % =============================================
%         function obj = ParInit(obj,Par)
%             % Init Par parameters
%             
%             if nargin < 2, error('Requires Par structure'); end
%             %global Par
%             
%             % init types internally
%             obj.GUI_STATES          = Par.GUI_STATES;
%             obj.BUTTON_TYPES        = Par.BUTTON_TYPES;
%             obj.ROI_TYPES           = Par.ROI_TYPES;
%             obj.IMAGE_TYPES         = Par.IMAGE_TYPES;
%             obj.ROI_AVERAGE_TYPES   = Par.ROI_AVERAGE_TYPES;
%             obj.VIEW_TYPES          = Par.VIEW_TYPES;
% 
%             % state of the ROI
%             obj.State               = obj.ROI_STATE_NONE;
%             
%             % ROI processing type
%             obj.AverType            = Par.ROI_AVERAGE_TYPES.MEAN;     % which operation to perform
%             obj.Color               = rand(1,3);       % generate colors
%             obj.Name                = 'X';             % which name
%             
%         end
        
        % =============================================
        function obj = Init(obj,roiType,viewType)
            % RoiInit - Init current/selected ROI main parameters. Not view/graphics related
            % Input
            %   roiType  - which ROI to create
            %   viewType  - which view ROI to create
            %   hAxes     - handle of the axes
            % Output
            %   obj     - updated
            
            if nargin < 2, roiType      = obj.ROI_TYPES.RECT; end;
            if nargin < 3, viewType     = obj.VIEW_TYPES.XY;   end;
            activeZstackIndex         = 1;   
            activeTimeIndex           = 1;   
            
            % check
            switch roiType,
                case {obj.ROI_TYPES.RECT,obj.ROI_TYPES.ELLIPSE,obj.ROI_TYPES.FREEHAND}
                otherwise error('Bad roiType')
            end
            switch viewType,
                case {obj.VIEW_TYPES.XY,obj.VIEW_TYPES.YT,obj.VIEW_TYPES.XYYT}
                otherwise error('Bad viewType')
            end
            
            
            % state of the ROI
            obj.State               = obj.STATE_TYPES.NONE;
            
            % ROI processing type
            obj.Type                = roiType;            % define ROI type
            obj.ViewType            = viewType; % main view type - where created
            obj.Color               = rand(1,3);       % generate colors
            obj.Name                = 'X';             % which name
            obj.zInd                = activeZstackIndex;           % location in Z stack
            obj.tInd                = activeTimeIndex;           % location in T stack
            obj.xyInd               = [];          % shape in xy plane
            obj.ytInd               = [];          % shape in xy plane
            obj.Data                = [];          % data process results

            %obj.SeqNum              = 1;           % sequence number TBD
            %obj.CountId             = 1;           % counter Id
            %obj.Position            = [0 0 1 1];   
            %obj.AverType            = 1; %obj.ROI_AVERAGE_TYPES.MEAN;     % which operation to perform
            %obj.CellPart            = 1; %obj.ROI_CELLPART_TYPES.ROI;
         
             % views
            obj.ViewXY               = [];                 
            obj.ViewYT               = [];
            
        end
        
        % =============================================
        function obj = InitView(obj,viewType,hAxes,Data)
            % InitView - Init current/selected ROI main parameters. Not view/graphics related
            % Input
            %   viewType  - which view of ROI to create
            %   hAxes     - handle of the axes
            %   Data      - starting point when Freehand is used
            % Output
            %   obj     - updated
            
            if nargin < 2, viewType     = obj.ViewType;   end;
            if nargin < 3, hAxes        = [];   end;
            if nargin < 4, Data         = 0;   end;
            
            
            % check
            if isempty(hAxes) || ~ishandle(hAxes)
                error('hAxes must be a valid handle')
            else
                axes(hAxes); % attention
            end
            
%             % in the fisrt time init
%             if obj.State == obj.STATE_TYPES.NONE,
%                 % init common ROI structure
%                 obj.Type     = roiType;            % define ROI type
%                 obj.State    = obj.STATE_TYPES.INIT;          % designates if this pointer structure is in use
%             else
%                 if obj.Type  ~= roiType,
%                     error('roiType on second init must match the first time')
%                 end
%             end,

            % get the view
            viewObj = GetView(obj,viewType);
            if ~isempty(viewObj),
                error('Already initalized')
            end


            % create View according to ROI type specified
            if obj.ViewType == viewType,
                viewObj         = InitShape(obj,Data);
            else
                tempViewType = obj.ViewType;
                obj.Type        = obj.ROI_TYPES.RECT;
                viewObj         = InitShape(obj,Data);
                obj.Type        = tempViewType;
            end
            
            % retutn back
            obj = SetView(obj,viewType,viewObj);            
            
            
%             % could be variations in XY
%             switch viewType,
%                 case obj.VIEW_TYPES.XY,
%                     % XY view allows different ROI forms
%                     obj.ViewType            = viewType; % main view type - where created
%                     switch obj.Type,
%                         case obj.ROI_TYPES.RECT,
%                             obj.ViewXY = InitShape(obj,0);
%                         case obj.ROI_TYPES.ELLIPSE,
%                             obj.ViewXY = InitShape(obj,0);
%                         case obj.ROI_TYPES.FREEHAND,
%                             obj.ViewXY = InitShape(obj,0);
%                             %obj.AverType = 'LineOrthog'; % props
%                             
%                         otherwise
%                             error('Type must be ROI_TYPE variable')
%                     end
%                     % state
%                     if obj.State == obj.STATE_TYPES.VIEWYT,
%                         obj.State    = obj.STATE_TYPES.VIEWALL;
%                     else
%                         obj.State    = obj.STATE_TYPES.VIEWXY;
%                     end
%                 case obj.VIEW_TYPES.YT,
%                     obj.Type        = obj.ROI_TYPES.RECT;            % define ROI type
%                     obj.ViewYT      = InitShape(obj,0);
%                     % state
%                     if obj.State == obj.STATE_TYPES.VIEWXY,
%                         obj.State    = obj.STATE_TYPES.VIEWALL;
%                     else
%                         obj.State    = obj.STATE_TYPES.VIEWYT;
%                     end
%                     
%                 otherwise
%                     error('viewType must be VIEW_TYPES variable')
%             end
        end
        
        % =============================================
        function viewObj = GetView(obj,viewType)
            % GetView - Get the relevant view
            % Input
            %   viewType  - which view of ROI to create
            % Output
            %   viewObj     - current view
            
            if nargin < 2, viewType     = obj.ViewType;   end;
            
            % get the view
            switch viewType,
                case obj.VIEW_TYPES.XY, 
                    viewObj = obj.ViewXY;
                case obj.VIEW_TYPES.YT,
                    viewObj = obj.ViewYT;
                otherwise
                    error('Bad viewType')
            end
            
        end
        
        % =============================================
        function obj = SetView(obj,viewType,viewObj)
            % SetView - Set the relevant view back
            % Input
            %   viewType  - which view of ROI to create
            %   viewObj     - current view
            % Output
            %   obj         - updated
            
            if nargin < 2, viewType     = obj.ViewType;   end;
            if nargin < 3, error('Must have view object');   end;
            
            
            % retutn back
            switch viewType,
                case obj.VIEW_TYPES.XY, 
                    obj.ViewXY = viewObj ;
                case obj.VIEW_TYPES.YT,
                    obj.ViewYT = viewObj ;
                otherwise
                    error('Bad viewType')
            end
            
        end
        
        % =============================================
        function isOK = CheckView(obj,viewType)
            % CheckView - Set the relevant view checks
            % Input
            %   viewType  - which view of ROI to create
            %   obj         - object with data
            % Output
            %   isOK         - false if view is bad
            
            if nargin < 2, viewType     = obj.ViewType;   end;
            isOK = false;
            
            % get
            viewObj = GetView(obj,viewType);    
            
            % checks
            if isempty(viewObj), return; end;
            if isempty(viewObj.hShape),return; end;
            if  ~ishandle(viewObj.hShape),return; end;
            
            isOK = true;
        end
        
        % =============================================
        function xy = PosToRect(obj,pos)
            % Helper function that transforms position to rectangles
            xy =  [ pos(1:2);...
                pos(1:2)+[0 pos(4)/2];...
                pos(1:2)+[0 pos(4)];...
                pos(1:2)+[pos(3)/2 pos(4)];...
                pos(1:2)+[pos(3) pos(4)];...
                pos(1:2)+[pos(3) pos(4)/2];...
                pos(1:2)+[pos(3) 0];...
                pos(1:2)+[pos(3)/2 0];...
                pos(1:2)];
            
        end
        
         % =============================================
        function ViewObj = InitShape(obj,Data)
            % Initalizes
            %global Par
            clr             = 'red';
            pos             = [0 0 .5 .5];
            posr            = PosToRect(obj,pos);
            curv            = [0,0]; % curvature
            switch obj.Type,
                case obj.ROI_TYPES.RECT,
                    % support xy shape input
                    xy               = repmat(pos(1:2),5,1) + [0 0;pos(3) 0;pos(3) pos(4); 0 pos(4);0 0];
                    if numel(Data) > 4, xy = Data; end;
                    pos              = [min(xy) max(xy) - min(xy)];
                case obj.ROI_TYPES.ELLIPSE,  % the same except curvature
                    % support xy shape input
                    xy               = repmat(pos(1:2),5,1) + [0 0;pos(3) 0;pos(3) pos(4); 0 pos(4);0 0];
                    if numel(Data) > 4, xy = Data; end;
                    pos              = [min(xy) max(xy) - min(xy)];
                    curv             = [1,1]; % curvature
                case obj.ROI_TYPES.FREEHAND,
                    if numel(Data) > 4,
                        xy           = Data;
                        pos          = [min(xy) max(xy) - min(xy)];
                    else
                        xy           = obj.pointRef(1,1:2);%hFree.getPosition;
                        pos          = pos + [xy 0 0];
                    end;
                otherwise
                    error('Type must be ROI_TYPE variable')
            end
            
            % init shape
            
            ViewObj.hShape  =  line('xdata',xy(:,1),'ydata',xy(:,2),...
                'lineStyle','--',...
                'lineWidth',1, ...
                'Color',clr);
           ViewObj.hCornRect  =  line('xdata',posr(:,1),'ydata',posr(:,2),...
               'lineStyle','--',...
               'Marker','s',...
               'MarkerSize',8,...
               'Color',clr);
            
            ViewObj.hBoundBox = rectangle('position',pos,...
                'lineStyle',':',...
                'lineWidth',0.5, ...
                'curvature', curv,... % added                
                'edgeColor',clr);
            ViewObj.hText      =  text(pos(1),pos(2),obj.Name,'color',obj.Color,'interpreter','none');
            
            %  hide it
            set(ViewObj.hShape,   'visible','off')
            set(ViewObj.hBoundBox,'visible','off')
            set(ViewObj.hCornRect,'visible','off')
            set(ViewObj.hText,    'visible','off')
            
            % save
            ViewObj.pos        = pos;
            ViewObj.posr       = posr;
            ViewObj.xy         = xy;
            ViewObj.clr        = clr;
            %View.name       = roiLast.Name;
            % add context menu
            % set(View.hBoundBox,'uicontextmenu',cntxMenu)
            
        end
         
        % =============================================
        function obj = AddPoint(obj,Data)
            % Adds data point to ROI shape.
            % Applicable only to XY view - freehand
            
            % add point to a freehand line
            newPoint                = Data;
            if isempty(newPoint), return; end;
            if obj.Type ~= obj.ROI_TYPES.FREEHAND,
                error('Should only be called in freehand object')
            end
            
            xData                   = [get(obj.ViewXY.hShape,'xdata') newPoint(1)];
            yData                   = [get(obj.ViewXY.hShape,'ydata') newPoint(2)];
            set(obj.ViewXY.hShape,'xdata',xData,'ydata',yData ,'color','b');
            
            % no scaling after position set
            obj.ViewXY.pos         = [min(xData) min(yData) max(xData)-min(xData) max(yData)-min(yData)];
            
            %  show it
            set(obj.ViewXY.hShape,   'visible','on')
        end
        
        % =============================================
        function obj = SetPosition(obj, Data, viewType)
            % define new ROI position
            % redraw the last ROI object pos
            %global Par;
            if nargin < 2, Data = [0 0]; end;
            if nargin < 3, viewType = obj.VIEW_TYPES.XY; end;
            
            pos                     = Data;
            switch viewType,
                case obj.VIEW_TYPES.XY,
                    viewObj          = obj.ViewXY;
%                     if obj.State ~= obj.STATE_TYPES.VIEWXY && obj.State ~= obj.STATE_TYPES.VIEWALL,
%                         error('XY not initiazed')
%                     end
                    switch obj.Type,
                        case obj.ROI_TYPES.RECT,
                            xy          = repmat(pos(1:2),5,1) + [0 0;pos(3) 0;pos(3) pos(4); 0 pos(4);0 0];
                        case obj.ROI_TYPES.ELLIPSE,
                            xyr         = pos(3:4)/2;        % radius
                            xyc         = pos(1:2)+ xyr;     % center
                            tt          = linspace(0,2*pi,30)';
                            xy          = [cos(tt)*xyr(1) + xyc(1), sin(tt)*xyr(2) + xyc(2)];
                        case obj.ROI_TYPES.FREEHAND,
                            pos_old     = obj.rectangleInitialPosition;
                            xy_old      = obj.shapeInitialDrawing;
                            % rescale
                            xyc         = pos(1:2)      + pos(3:4)/2;     % center
                            xyc_old     = pos_old(1:2)  + pos_old(3:4)/2;     % center
                            x           = (xy_old(:,1) - xyc_old(1))*pos(3)/(eps+pos_old(3)) + xyc(1);
                            y           = (xy_old(:,2) - xyc_old(2))*pos(4)/(eps+pos_old(4)) + xyc(2);
                            xy          = [x(:) y(:)];
                    end;
                case obj.VIEW_TYPES.YT,
%                     if obj.State ~= obj.STATE_TYPES.VIEWYT || obj.State ~= obj.STATE_TYPES.VIEWALL,
%                         error('YT not initiazed')
%                     end
                    xy          = repmat(pos(1:2),5,1) + [0 0;pos(3) 0;pos(3) pos(4); 0 pos(4);0 0];
                    viewObj          = obj.ViewYT;
                otherwise
                    error('Bad viewType')
            end
            
            viewObj.pos             = pos; %[x,y,w,h]
            posr                    = obj.PosToRect(pos);
            
            % rect to xy
            set(viewObj.hShape,     'xdata',xy(:,1),'ydata',xy(:,2),'visible','on');
            set(viewObj.hBoundBox,  'position',pos,'visible','on');
            set(viewObj.hCornRect,  'xdata',posr(:,1),'ydata',posr(:,2),'visible','on' );
            set(viewObj.hText,       'pos',pos(1:2)+[5,5], 'visible','on');
            
            % save back
            switch viewType,
                case obj.VIEW_TYPES.XY,
                    obj.ViewXY = viewObj;
                case obj.VIEW_TYPES.YT,
                    obj.ViewYT = viewObj;
            end
            
        end
        
        % =============================================
        function Pos = GetPosition(obj, viewType)
            % Get position from the view
            if nargin < 2, viewType = obj.VIEW_TYPES.XY; end;
            
            viewObj         = GetView(obj,viewType);
            Pos             = viewObj.pos;
            
        end
        
        % =============================================
        function obj = SetColor(obj, Data)
            % set color for all views
            %global Par;
            if nargin < 2, Data = [0 0]; end;
            
            clr                 = Data;
            % check init
            if obj.State == obj.STATE_TYPES.VIEWXY || obj.State == obj.STATE_TYPES.VIEWALL,
                obj.Color           = clr;   % remember
                set(obj.ViewXY.hShape,     'Color',    clr);
                set(obj.ViewXY.hBoundBox,  'edgeColor',clr);
                set(obj.ViewXY.hCornRect,  'Color',    clr);
                set(obj.ViewXY.hText,      'color',    clr);
            end;
            
            % check init
            if obj.State == obj.STATE_TYPES.VIEWYT || obj.State == obj.STATE_TYPES.VIEWALL,
                obj.Color           = clr;   % remember
                set(obj.ViewYT.hShape,     'Color',    clr);
                set(obj.ViewYT.hBoundBox,  'edgeColor',clr);
                set(obj.ViewYT.hCornRect,  'Color',    clr);
                set(obj.ViewYT.hText,      'color',    clr);
            end;
        end
        
        % =============================================
        function obj = SetName(obj, Data)
            % set name of the ROI
            % redraw the last ROI object pos
            assert(ischar(Data), 'Data Must be char');
            
            obj.Name        = Data;   % remember
            set(obj.ViewXY.hText,  'string', obj.Name,   'visible','on')
        end
       
        % =============================================
        function obj = SetZInd(obj, Data)
            % set z index
            obj.zInd        = Data;
        end
        
        % =============================================
        function obj = SetTInd(obj, Data)
            % set z index
            obj.tInd        = Data;
        end
        
        % =============================================
        function obj = SaveInitRef(obj, Data)
        % remember ref position        
            % help varibles
            obj.rectangleInitialPosition    = Data; %RoiLast.Position;
            obj.shapeInitialDrawing         = [get(obj.ViewXY.hShape,'xdata')' get(obj.ViewXY.hShape,'ydata')']; 
        end

        % =============================================
        function [obj,isOK] = ImportRoi(obj, strROI)
        % init ROI from outside structure        
            % check the XY size only if less than 7x7 pix - kill
            if nargin < 2, error('strROI required'); end;
            
            %global Par
            thisRoiAreaOK                 = true;   
            thisRoiNeedsInit              = true;
            isOK                          = true;
            
            % support roi types
            if isprop(strROI,'Type'),
                thisRoiNeedsInit         = false;
            end;
            
            % support count id
            if isprop(strROI,'CountId'),
                obj.CountId  = strROI.CountId;
            else
                obj.CountId  = 1;
            end;
            
            currentXY                   = [1 1; 1 3;3 3;3 1; 1 1];
            if isprop(strROI,'xyInd')
                currentXY               = strROI.xyInd;
            else
                thisRoiAreaOK           = false;
            end
            
            % check area
            rectArea                    = prod(max(currentXY) - min(currentXY));
            if rectArea < 10,
                thisRoiAreaOK             = false;
            end
            
            % check max Id            
            if thisRoiAreaOK , 
                
                if thisRoiNeedsInit,
                    obj.Type        = obj.ROI_TYPES.ELLIPSE;
                    obj.CountId     = 1;
                    obj.ViewXY       = InitShape(obj,currentXY);
                    obj              = SetName(obj, 'Change My Name');
%                     fManageLastRoi('initFreehand',currentXY);
%                     fManageLastRoi('setName',sprintf('ROI:%2d Z:%d',i,activeZstackIndex));
                else                
                    obj              = strROI; 
                    obj.ViewXY       = InitShape(obj,0);
               end
                
                % add to the pool
                %fManageRoiArray('copyToList',i)
                
            else            
                %fManageRoiArray('delete',i)
                isOK = false;
                
           end
        end
        
        % =============================================
        function [obj, isOK] = DeleteView(obj,viewType)
            % DeleteView - delete certain view
            % Input
            %   viewType  - which view of ROI to create
            %   obj         - object with data
            % Output
            %   obj         - is updated
            %   isOK        - false if fails
            
            if nargin < 2, viewType     = obj.ViewType;   end;
            
            % check
            isOK    = CheckView(obj,viewType);
            if ~isOK, return; end;
            
            % get
            viewObj = GetView(obj,viewType);  
            
            % save the data
            xy      = [get(viewObj.hShape,'xdata')' get(viewObj.hShape,'ydata')'];
            switch viewType,
                case obj.VIEW_TYPES.XY, obj.xyInd = xy;
                case obj.VIEW_TYPES.YT, obj.ytInd = xy;
            end
            
            % remove graphics
            delete(viewObj.hShape);
            delete(viewObj.hBoundBox);
            delete(viewObj.hCornRect);
            delete(viewObj.hText);
            viewObj = [];
            
            % return
            obj     = SetView(obj,viewType,viewObj);
            isOK    = true;
        end
        
        % =============================================
        function obj = Delete(obj)
            % redraw the last ROI object pos
            % delete graphics of the lastROI
            %if activeRectangleIndex < 1, return; end;
            
            
            %if obj.State == obj.STATE_TYPES.VIEWXY || obj.State == obj.STATE_TYPES.VIEWALL,
            if ~isempty(obj.ViewXY)
                delete(obj.ViewXY.hShape);
                delete(obj.ViewXY.hBoundBox);
                delete(obj.ViewXY.hCornRect);
                delete(obj.ViewXY.hText);
                obj.ViewXY = [];
            end;
            
            % check init
            %if obj.State == obj.STATE_TYPES.VIEWYT || obj.State == obj.STATE_TYPES.VIEWALL,
            if ~isempty(obj.ViewYT)
                delete(obj.ViewYT.hShape);
                delete(obj.ViewYT.hBoundBox);
                delete(obj.ViewYT.hCornRect);
                delete(obj.ViewYT.hText);
                obj.ViewYT = [];
            end;
        end
        
        % =============================================
        function obj = UpdateView(obj)

                % redraw XY view according to info from YT view
                
                % check if YT is initialized
                if ~isfield(obj.ViewYT,'hBoundBox'), return; end;
                
                % extract Y length from YT space
                posXY                = get(obj.ViewXY.hBoundBox,'pos');
                posYT                = get(obj.ViewYT.hBoundBox,'pos');
                
                % position is defined by 
                posXY([2 4])         = posYT([2 4]);
                
                % redefine the shape
                obj                  = SetPosition(obj, posXY);
                % update color to default
                obj                  = SetColor(obj, obj.Color);
        end

        % =============================================
        function obj = CreateView(obj)

                % redraw XY view according to info from YT view
                if ~isfield(obj.ViewYT,'hBoundBox'), return; end;
                
                % extract Y length from YT space
                posYT                = get(obj.ViewYT.hBoundBox,'pos');
                
                % position is defined by 
                posXY([2 4])         = posYT([2 4]);
                posXY([1 3])         = [10 nC-20];    
                
               switch obj.Type,
                     case obj.ROI_TYPES.ELLIPSE,
                        curv           = [1,1]; % curvature
                    otherwise
                        curv           = [0,0]; % curvature
                end;
                
                
                % init shapes
                pos                  = posXY;
                xy                   = repmat(pos(1:2),5,1) + [0 0;pos(3) 0;pos(3) pos(4); 0 pos(4);0 0];                
                clr                 = 'y';
                obj.ViewXY.hShape  =  line('xdata',xy(:,1),'ydata',xy(:,2),...
                        'lineStyle','--',...
                        'lineWidth',1, ...
                        'Color',clr);
                obj.ViewXY.hBoundBox = rectangle('position',pos,...
                        'lineStyle',':',...
                        'lineWidth',0.5, ...
                        'curvature',curv,...                       
                        'edgeColor',clr);
                cornerRectangles = getCornerRectangles(pos);
                for j=1:8,
                 obj.ViewXY.hCornRect(j) = rectangle('position',cornerRectangles{j},...
                        'lineWidth',1, ...
                        'edgeColor',clr);
                end
                obj.ViewXY.hText =  text(pos(1),pos(2),roiLast.Name,'color',roiLast.Color,'interpreter','none');  
                
                %  hide it
                set(obj.ViewXY.hShape,   'visible','off')
                set(obj.ViewXY.hBoundBox,'visible','off')
                set(obj.ViewXY.hCornRect,'visible','off')
                set(obj.ViewXY.hText,    'visible','off')
                
                % add context menu
                 set(obj.ViewXY.hBoundBox,'uicontextmenu',cntxMenu)
                
                % redefine the shape
                obj                  = SetPosition(obj, posXY);
                % update color to default
                obj                  = SetColor(obj, obj.Color);
        
                
        end
        
        % =============================================
        function obj = SetActive(obj,clr,viewType)
            % SetActive - activates current rectngle
            if nargin < 2, clr = 'r'; end;
            if nargin < 3, viewType = obj.ViewType; end;
            
            % check colors
            switch clr,
                case {'y','r','b'},
                otherwise error('clr must be y,r,b')
            end
            
            % get the view
            viewObj = GetView(obj,viewType);            
            assert(~isempty(viewObj),'View Must be initalized');
                
            % which object type
           switch obj.Type,
                 case obj.ROI_TYPES.ELLIPSE,
                    curv           = [1,1]; % curvature
                otherwise
                    curv           = [0,0]; % curvature
            end;

            
            set(viewObj.hShape,     'color',    clr,'visible','on');
            set(viewObj.hBoundBox,  'edgeColor',clr,'visible','on', 'curvature', curv);
            set(viewObj.hCornRect,  'Color',    clr,'visible','on');
            set(viewObj.hText,      'color',    clr,'visible','on')

            % gui support
            obj.rectangleInitialPosition    = viewObj.pos;
            obj.shapeInitialDrawing         = [get(viewObj.hShape,'xdata')' get(viewObj.hShape,'ydata')']; 
            
            % get back    
            obj = SetView(obj,viewType,viewObj);
                
        end
        
        % =============================================
        function obj = SetNonActive(obj,viewType)
            % SetNonActive - dis-activates current rectngle
            if nargin < 2, viewType = obj.ViewType; end;
            
            % get the view
            viewObj = GetView(obj,viewType);            
            assert(~isempty(viewObj),'View Must be initalized');
                
            % which object type
            clr                     = 'y';
            set(viewObj.hShape,     'Color',      clr,'visible','on');
            set(viewObj.hBoundBox,  'edgeColor',  clr,'visible','off', 'curvature', [0,0]);
            set(viewObj.hCornRect,  'Color',      clr,'visible','off');
            set(viewObj.hText,      'color',      clr,'visible','on')
            
            % get back    
            obj = SetView(obj,viewType,viewObj);
                
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function obj = fCreateRoiContextMenu(obj)
            % right click menu
            obj.cntxMenu = uicontextmenu;
            uimenu(obj.cntxMenu,'Label','Remove ROI',    'Callback',         @fRemoveMarkedRoi);
            uimenu(obj.cntxMenu,'Label','Select Color',  'Callback',         @fSelectColorForRoi);
            uimenu(obj.cntxMenu,'Label','Rename',        'Callback',         @fRenameRoi);
            uimenu(obj.cntxMenu,'Label','Aver Type',     'Callback',         @fAverageTypeRoi);
            uimenu(obj.cntxMenu,'Label','Cell Part Type','Callback',         @fCellPartTypeRoi);
            uimenu(obj.cntxMenu,'Label','Show name',     'Callback',         @fShowNameRoi, 'checked', 'off');
            uimenu(obj.cntxMenu,'Label','Snap to Data',  'Callback',         'warndlg(''TBD'')');
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = TestUserInput(obj,a,b)
            
            %motionClickWithLeftClick = true;
            units = get(obj.hAxes,'units');
            set(obj.hAxes,'units','normalized');
            point2 = get(obj.hAxes, 'CurrentPoint');
            set(obj.hAxes,'units',units);
            
            %fRefreshImage (); %plot the image
            %obj.AddPoint(point2(1,1:2));
            obj = obj.SetPosition( [point2(1,1:2) 20 20]);            
            
            obj.SetColor('blue');
            
        end
        
        % =============================================
        function obj = TestSimple(obj)
            % TestSimple - create image connect callback and init an object
            
            % GUI
            obj.hFigure = figure(10); clf; set(obj.hFigure,'Position',[100 100 560 580]);
            obj.hAxes   = axes('DataAspectRatio',[1 1 1],'Parent',obj.hFigure);
            obj.hImage = image(imread('cell.tif'),'CDataMapping','scaled','Parent',obj.hAxes);
            
            % init ROI
            obj = obj.Init(obj.ROI_TYPES.FREEHAND,obj.VIEW_TYPES.XY);
            obj = obj.InitView(obj.VIEW_TYPES.XY,obj.hAxes);
            obj = obj.SetPosition( [10 10 20 20], obj.VIEW_TYPES.XY);
            obj = obj.SetColor( [0 0 0.5]);
            %obj = obj.Delete();
            
            %set(seal.Figure,'WindowButtonMotion',@(src,event)TestUserInput(obj, src, eventdata, seal));
            set(obj.hFigure,'WindowButtonDown',@(src,event)TestUserInput(obj, src, event));
            
            
        end % TestSimple
        
        
    end% methods
end% classdef
