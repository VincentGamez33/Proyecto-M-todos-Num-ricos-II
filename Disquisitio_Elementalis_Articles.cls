\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Disquisitio_Elementalis_Articles}[2025/02/16 Clase personalizada basada en 'extarticle' para LuaLaTeX]

\RequirePackage{expl3}

\LoadClass[twoside,9pt]{extarticle}

\RequirePackage[spanish,es-noshorthands,es-tabla,es-lcroman]{babel}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Paquetes Necesarios %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------
% Paquetes Fundamentales y de Programación
% --------------------------------------------------------
\RequirePackage{ifthen}          % Proporciona sentencias condicionales.
\RequirePackage{etoolbox}        % Herramientas avanzadas de programación para LaTeX.
\RequirePackage{xparse}          % Análisis avanzado de argumentos para macros.
\RequirePackage{calc}            % Notación infija para aritmética.
\RequirePackage{silence}         % Suprimir advertencias específicas.
    \WarningsOff[blindtext]      % Ejemplo: Suprimir advertencias del paquete blindtext.
    \WarningFilter{latex}{Command \showhyphens has changed}
\RequirePackage{fancyvrb}

% --------------------------------------------------------
% Figuras, Imágenes y Cajas de Color
% --------------------------------------------------------
\RequirePackage{graphicx}                          % Para incluir y manipular imágenes.
\RequirePackage{caption}                           % Personalización de títulos de flotantes (figuras, tablas).
\RequirePackage{chngcntr}
\RequirePackage{subcaption}                        % Subfiguras y subtítulos.
\RequirePackage{floatrow}                          % Personalización avanzada del layout de flotantes.
\RequirePackage[section]{placeins}                 % Evita que los flotantes traspasen las barreras de sección.
\RequirePackage[dvipsnames,svgnames,table]{xcolor} % Soporte extendido para colores.
    
\RequirePackage[framemethod=TikZ]{mdframed}        % Cajas enmarcadas personalizables.
\RequirePackage[most]{tcolorbox}                   % Cajas de colores y resaltadas muy potentes.
    \tcbuselibrary{skins}
    \tcbuselibrary{listings,breakable}
    \tcbuselibrary{documentation}
    \tcbuselibrary{theorems}
    \tcbuselibrary{hooks}

\RequirePackage{fancyhdr}

\RequirePackage{datetime}

\RequirePackage{lipsum}

\RequirePackage{titlesec}

\RequirePackage{hyperref}

% --------------------------------------------------------
% Gráficos Vectoriales con TikZ y PGFPlots
% --------------------------------------------------------
\RequirePackage{tikz}
\RequirePackage{pgfplots}
    \pgfplotsset{compat=1.18}
\RequirePackage{circuitikz}    % Para dibujar circuitos eléctricos.
\RequirePackage{tikz-3dplot}   % Gráficos 3D con TikZ.
\RequirePackage{tikzpagenodes} % Nodos de página para posicionamiento con TikZ.
    % --- Librerías de TikZ ---
    \usetikzlibrary{shapes, shapes.geometric}
    \usetikzlibrary{positioning, calc, backgrounds, tikzmark, fit}
    \usetikzlibrary{calendar, mindmap, trees}
    \usetikzlibrary{arrows.meta, shadings, shadows}
    \usetikzlibrary{intersections, decorations.pathreplacing, decorations.markings, decorations}
    \usetikzlibrary{matrix, 3d, patterns, angles, quotes, math}
    \usetikzlibrary{pgfplots.fillbetween}

\RequirePackage{ragged2e}

% --------------------------------------------------------
% Tablas
% --------------------------------------------------------
\RequirePackage{tabularx}   % Tablas de ancho flexible.
\RequirePackage{nicematrix} % Creación de tablas complejas con capacidades PGF/TikZ.
\RequirePackage{xltabular}  % Combina tabularx y longtable.
\RequirePackage{booktabs}   % Estilo profesional para líneas en tablas.
\RequirePackage{multirow}   % Celdas que abarcan varias filas.
\RequirePackage{longtable}  % Tablas que pueden dividirse entre páginas.
\RequirePackage{makecell}   % Saltos de línea y formato mejorado dentro de las celdas.
\RequirePackage{threeparttable}

\RequirePackage{multicol}

% --------------------------------------------------------
% Matemáticas y Símbolos
% --------------------------------------------------------
\PassOptionsToPackage{warnings-off={mathtools-colon,mathtools-overbracket}}{unicode-math}
\RequirePackage{amsmath,amssymb,amsthm} % Símbolos matemáticos extendidos (AMS), carga amsfonts.
\RequirePackage{mathtools}
\RequirePackage{cancel}                 % Para tachar expresiones matemáticas.
\RequirePackage{bm}                     % Símbolos en negrita.
\RequirePackage{thmtools}               % Herramientas para entornos de teoremas.
\RequirePackage{siunitx}                % Composición de unidades del Sistema Internacional.

\RequirePackage{marginnote} % Notas al margen.
\RequirePackage{marginfix}  % Corrige problemas de posicionamiento de notas al margen.
\RequirePackage{sidenotes}  % Entorno para notas al margen.

\RequirePackage{enumitem}

\RequirePackage[
    backend=biber,
    style=nature,
]{biblatex}

% --------------------------------------------------------
% Paquetes personalizados.
% --------------------------------------------------------
\RequirePackage{Setup/00-Fonts}
\RequirePackage{Setup/01-Colors}
\RequirePackage{Setup/02-Geometry}
\RequirePackage{Setup/03-References}
\RequirePackage{Setup/04-Headings}
\RequirePackage{Setup/05-FootNotes}
\RequirePackage{Setup/06-Sections}
\RequirePackage{Setup/07-Floating}
\RequirePackage{Setup/08-Math}
\RequirePackage{Setup/09-Title}
\RequirePackage{Setup/10-Code}

\AtBeginDocument{
    \pagestyle{fancy}
}