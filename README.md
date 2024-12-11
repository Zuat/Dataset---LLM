Modelo y Arquitectura de Chatbot Basado en LLM
Este capítulo explica cómo funciona el modelo utilizado en el proyecto, su propósito, los componentes clave, y cómo se entrena para garantizar una comprensión amplia, tanto para informáticos como para no especialistas.

Definición del Modelo
El chatbot desarrollado en este proyecto utiliza Large Language Models (LLM), específicamente el modelo Sentence-BERT (SBERT), para proporcionar respuestas automatizadas basadas en la similitud semántica entre el mensaje del usuario y un conjunto de intenciones predefinidas. Este enfoque permite una interacción precisa y natural con los usuarios, optimizando procesos de soporte y orientación en tareas de diseño arquitectónico de software.

En esta imagen podemos ver cómo se describen y funcionan las sentencias en SBERT, donde cada sentencia (en este caso u o v) es clasificada con los embeddings para el entrenamiento y "la similitud del coseno" para las inferencias que detonan la respuesta del prototipo del chatbot (Kotamraju, 2022).

¿Qué hace el modelo?
Conversión de texto a vectores numéricos (embeddings):
SBERT convierte frases o mensajes en representaciones matemáticas llamadas embeddings. Estas son vectores de alta dimensionalidad que encapsulan el significado semántico del texto.

Medición de similitud semántica:
Cuando el usuario envía un mensaje, SBERT genera un embedding correspondiente. Este se compara con embeddings precalculados para determinar qué intención es más similar al mensaje del usuario. La similitud se calcula mediante una métrica conocida como cosine similarity.

Selección de respuesta:
Si la similitud es suficientemente alta (supera un umbral definido, como 0.55), el chatbot selecciona una respuesta predefinida asociada a la intención más relevante. Si no alcanza el umbral, solicita una reformulación del mensaje.

Componentes del Modelo
Carga y procesamiento de datos:

Dataset: El archivo datasetPT.json contiene patrones de frases (mensajes representativos de los usuarios) y respuestas asociadas.
Preprocesamiento: Los patrones se limpian eliminando stopwords (palabras irrelevantes como "el", "de", "y") y aplicando lematización (reducir palabras a su forma base). Esto reduce el ruido y mejora la precisión.
Sentence-BERT:
SBERT se utiliza para convertir patrones y mensajes en embeddings. Estos embeddings permiten comparar frases basándose en su significado, no solo en palabras exactas.

Similitud de Coseno:

Mide el ángulo entre dos vectores en un espacio multidimensional.
Un valor cercano a 1 indica alta similitud, mientras que valores más bajos sugieren mensajes diferentes.
Sistema de respuesta:

Integra un umbral dinámico para determinar si una intención coincide con el mensaje del usuario.
Proporciona respuestas predefinidas o solicita aclaraciones.
¿Cómo se entrena Sentence-BERT?
SBERT se entrena en dos fases clave, que lo convierten en una herramienta poderosa para comparar y generar representaciones semánticas de frases.

Pre-entrenamiento:

Propósito: Adquirir una comprensión general del lenguaje.
Técnica: Se entrena en tareas como la predicción de palabras faltantes y el análisis de contexto usando grandes cantidades de texto no etiquetado.
Fine-tuning (Ajuste fino):

Propósito: Adaptar SBERT a tareas específicas, como la comparación semántica de frases.
Técnica: Usa conjuntos de datos etiquetados con pares de oraciones y su nivel de similitud. Esto permite al modelo aprender a generar embeddings que representen fielmente el significado de frases completas.
Optimización durante el ajuste fino:

Función de pérdida: Utiliza técnicas como contrastive loss, que maximizan la similitud entre frases relacionadas y minimizan la similitud entre frases no relacionadas.
Red neuronal siamesa: Dos instancias idénticas del modelo generan embeddings para cada frase por separado, optimizando la comparación directa entre ellas.
Arquitectura de Sentence-BERT
SBERT mejora la arquitectura de BERT tradicional mediante redes neuronales siamesas, que permiten generar embeddings independientes para cada frase.

Entrada:

Fragmenta las frases en tokens utilizando el método WordPiece. Esto divide palabras complejas en partes más manejables.
Capa de embeddings:

Convierte los tokens en vectores que representan su significado semántico.
Capa de pooling:

Promedia los embeddings de todos los tokens de una frase para crear un vector único que capture el significado completo.
Cálculo de similitud:

Compara embeddings usando similitud de coseno, optimizando el tiempo de inferencia en tareas de comparación.
Importancia del Modelo
Entendimiento del lenguaje humano:
SBERT no solo analiza palabras, sino que comprende el contexto completo de frases, facilitando interacciones más naturales.

Eficiencia en tareas semánticas:
Reduce el tiempo de procesamiento comparado con BERT tradicional, lo que lo hace ideal para aplicaciones en tiempo real como chatbots.

Flexibilidad en aplicaciones:
Desde sistemas de recomendación hasta análisis de sentimientos, su capacidad de capturar relaciones semánticas lo convierte en una herramienta versátil.
