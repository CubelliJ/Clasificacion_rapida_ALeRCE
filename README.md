# Clasificacion_rapida_ALeRCE
Clasificación rápida en ALeRCE utilizando selección de características y algoritmos genéticos.

El objetivo de este proyecto es reducir el número de características utilizadas por el clasificador de curvas de luz de ALeRCE, explorando el compromiso entre el desempeño en clasificación y el costo computacional asociado.

La primera etapa del proyecto consiste en evaluar la estrategia greedy para seleccionar las características más relevantes al momento de clasificar. En esta estrategia se explorará una característica a la vez, buscando cuál de ellas ayuda más al desempeño del modelo cuando es agregada al conjunto de características utilizadas. También se deberá probar comenzando con todas las características y removiendo una característica a la vez. Notar que en esta etapa no se toma en cuenta explícitamente el costo de computar cada característica al momento de efectuar la selección.
