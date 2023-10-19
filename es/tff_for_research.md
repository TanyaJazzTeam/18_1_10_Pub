# Uso de TFF para la investigación del aprendizaje federado

## Resumen ds

# klo

![Ciudadela](https://vignette.wikia.nocookie.net/masseffect/images/d/d7/MassEffect2Citadel.jpg/revision/latest?cb=20100721191415)

TFF es un marco potente y extensible para realizar investigaciones de aprendizaje federado (FL) mediante la simulación de cálculos federados en conjuntos de datos proxy realistas. Esta página describe los conceptos y componentes principales que son relevantes para las simulaciones de investigación, así como también una guía detallada para realizar diferentes tipos de investigación en TFF.

## La estructura típica del código de investigación en TFFdearf лшг

Una simulación de FL de investigación implementada en TFF normalmente consta de tres tipos principales de lógica.

1. Piezas individuales de código de TensorFlow, normalmente `tf.function` , que encapsulan la lógica que se ejecuta en una única ubicación (por ejemplo, en clientes o en un servidor). Este código normalmente se escribe y prueba sin ninguna referencia `tff.*` y se puede reutilizar fuera de TFF. Por ejemplo, el [ciclo de capacitación del cliente en Promedio federado](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) se implementa en este nivel.
2. Lógica de orquestación federada de TensorFlow, que une los `tf.function` individuales de 1. envolviéndolos como `tff.tf_computation` sy luego orquestándolos usando abstracciones como `tff.federated_broadcast` y `tff.federated_mean` dentro de un `tff.federated_computation` . Consulte, por ejemplo, esta [orquestación para el promedio federado](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140) .
3. Un script de controlador externo que simula la lógica de control de un sistema FL de producción, seleccionando clientes simulados de un conjunto de datos y luego ejecutando cálculos federados definidos en 2. en esos clientes. Por ejemplo, [un controlador de experimento EMNIST federado](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) .

## Conjuntos de datos de aprendizaje federado ewt

TensorFlow federado [aloja múltiples conjuntos de datos](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets) que son representativos de las características de los problemas del mundo real que podrían resolverse con el aprendizaje federado.

Nota: Estos conjuntos de datos también pueden ser consumidos por cualquier marco de aprendizaje automático basado en Python como matrices Numpy, como se documenta en la [API ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData) . ewt

Los conjuntos de datos incluyen:

- [**Desbordamiento de pila** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) Un conjunto de datos de texto realista para modelado de lenguaje o tareas de aprendizaje supervisado, con 342.477 usuarios únicos con 135.818.730 ejemplos (oraciones) en el conjunto de entrenamiento.
- [**EMNIST Federado** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) Un preprocesamiento federado del conjunto de datos de caracteres y dígitos de EMNIST, donde cada cliente corresponde a un escritor diferente. El conjunto completo de trenes contiene 3.400 usuarios con 671.585 ejemplos de 62 etiquetas.
- [**Shakespeare** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data) Un conjunto de datos de texto más pequeño a nivel de caracteres basado en las obras completas de William Shakespeare. El conjunto de datos consta de 715 usuarios (personajes de obras de Shakespeare), donde cada ejemplo corresponde a un conjunto contiguo de líneas pronunciadas por el personaje de una obra determinada.
- [**CIFAR-100** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) Una partición federada del conjunto de datos CIFAR-100 en 500 clientes de entrenamiento y 100 clientes de prueba. Cada cliente tiene 100 ejemplos únicos. La partición se realiza de manera que se cree una heterogeneidad más realista entre los clientes. Para obtener más detalles, consulte la [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) .
- [**Conjunto de datos de Google Landmark v2** El conjunto de datos consta de fotografías de varios lugares emblemáticos del mundo, con imágenes agrupadas por fotógrafo para lograr una partición federada de los datos. Hay dos tipos de conjuntos de datos disponibles: un conjunto de datos más pequeño con 233 clientes y 23080 imágenes, y un conjunto de datos más grande con 1262 clientes y 164172 imágenes.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data)
- [**CelebA** Un conjunto de datos de ejemplos (imagen y atributos faciales) de rostros de celebridades. El conjunto de datos federado tiene los ejemplos de cada celebridad agrupados para formar un cliente. Hay 9343 clientes, cada uno con al menos 5 ejemplos. El conjunto de datos se puede dividir en grupos de entrenamiento y prueba, ya sea por clientes o por ejemplos.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data)
- [**iNaturalist** Un conjunto de datos consta de fotografías de varias especies. El conjunto de datos contiene 120.300 imágenes de 1.203 especies. Hay siete versiones del conjunto de datos disponibles. Uno de ellos está agrupado por el fotógrafo y consta de 9257 clientes. El resto de los conjuntos de datos están agrupados por la ubicación geográfica donde se tomó la foto. Estos seis tipos del conjunto de datos constan de 11 a 3606 clientes.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data)

## Simulaciones de alto rendimiento etw

Si bien el tiempo de una *simulación* de FL no es una métrica relevante para evaluar algoritmos (ya que el hardware de simulación no es representativo de los entornos de implementación de FL reales), poder ejecutar simulaciones de FL rápidamente es fundamental para la productividad de la investigación. Por lo tanto, TFF ha invertido mucho en proporcionar tiempos de ejecución de alto rendimiento para una o varias máquinas. La documentación está en desarrollo, pero por ahora consulte las instrucciones sobre [simulaciones de TFF con aceleradores](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators) y las instrucciones sobre [cómo configurar simulaciones con TFF en GCP](https://www.tensorflow.org/federated/gcp_setup) . El tiempo de ejecución TFF de alto rendimiento está habilitado de forma predeterminada.

## TFF para diferentes áreas de investigación

### Algoritmos de optimización federados

La investigación sobre algoritmos de optimización federados se puede realizar de diferentes maneras en TFF, según el nivel de personalización deseado.

[Aquí](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg) se proporciona una implementación mínima e independiente del algoritmo [de promedio federado](https://arxiv.org/abs/1602.05629) . El código incluye [funciones TF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py) para cálculo local, [cálculos TFF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) para orquestación y un [script de controlador](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) en el conjunto de datos EMNIST como ejemplo. Estos archivos se pueden adaptar fácilmente para aplicaciones personalizadas y cambios algorítmicos siguiendo instrucciones detalladas en el [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md) .

Puede encontrar una implementación más general del promedio federado [aquí](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py) . Esta implementación permite técnicas de optimización más sofisticadas, incluido el uso de diferentes optimizadores tanto en el servidor como en el cliente. Otros algoritmos de aprendizaje federado, incluida la agrupación en clústeres de k-medias federada, se pueden encontrar [aquí](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/) .drwrwrf wrfrf

### Compresión de actualización del modelo

La compresión con pérdidas de las actualizaciones del modelo puede generar costos de comunicación reducidos, lo que a su vez puede generar una reducción del tiempo general de capacitación.

Para reproducir un [artículo](https://arxiv.org/abs/2201.02664) reciente, consulte [este proyecto de investigación](https://github.com/google-research/federated/tree/master/compressed_communication) . Para implementar un algoritmo de compresión personalizado, consulte [los métodos de comparación](https://github.com/google-research/federated/tree/master/compressed_communication/aggregators/comparison_methods) en el proyecto para obtener líneas base como ejemplo y [el tutorial de agregadores de TFF](https://www.tensorflow.org/federated/tutorials/custom_aggregators) si aún no está familiarizado con ellos.

### Privacidad diferencial

TFF es interoperable con la biblioteca [de privacidad de TensorFlow](https://github.com/tensorflow/privacy) para permitir la investigación de nuevos algoritmos para el entrenamiento federado de modelos con privacidad diferencial. Para ver un ejemplo de entrenamiento con DP utilizando [el algoritmo básico DP-FedAvg](https://arxiv.org/abs/1710.06963) y [sus extensiones](https://arxiv.org/abs/1812.06210) , consulte [este controlador de experimento](https://github.com/google-research/federated/blob/master/differential_privacy/stackoverflow/run_federated.py) .

Si desea implementar un algoritmo DP personalizado y aplicarlo a las actualizaciones agregadas del promedio federado, puede implementar un nuevo algoritmo de media DP como una subclase de [](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) `tensorflow_privacy.DPQuery` y cree un `tff.aggregators.DifferentiallyPrivateFactory` con una instancia de su consulta. Se puede encontrar un ejemplo de implementación del [algoritmo DP-FTRL.](https://arxiv.org/abs/2103.00039) [](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

Las GAN federadas (descritas [a continuación](#generative_adversarial_networks) ) son otro ejemplo de un proyecto TFF que implementa privacidad diferencial a nivel de usuario (por ejemplo, [aquí en el código](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L144) ).

### Robustez y ataques

TFF también se puede utilizar para simular los ataques dirigidos a sistemas de aprendizaje federados y defensas diferenciales basadas en la privacidad consideradas en *[¿Puede realmente el aprendizaje federado por la puerta trasera?](https://arxiv.org/abs/1911.07963) . Esto se hace mediante la creación de un proceso iterativo con clientes potencialmente maliciosos (consulte [](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L412) `build_federated_averaging_process_attacked` ). El [directorio contiene más detalles.](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack)*

- Se pueden implementar nuevos algoritmos de ataque escribiendo una función de actualización del cliente que sea una función de Tensorflow; consulte [`ClientProjectBoost` para ver un ejemplo.](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460)
- Se pueden implementar nuevas defensas personalizando ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103) , que agrega resultados del cliente para obtener una actualización global.

Para ver un script de ejemplo para simulación, consulte [`emnist_with_targeted_attack.py` .](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/emnist_with_targeted_attack.py)

### Redes generativas de confrontación

Las GAN crean un [patrón de orquestación federada](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L266-L316) interesante que se ve un poco diferente al promedio federado estándar. Implican dos redes distintas (el generador y el discriminador), cada una entrenada con su propio paso de optimización.

TFF se puede utilizar para investigaciones sobre capacitación federada de GAN. Por ejemplo, el algoritmo DP-FedAvg-GAN presentado en [un trabajo reciente](https://arxiv.org/abs/1911.06679) se [implementa en TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans) . Este trabajo demuestra la eficacia de combinar aprendizaje federado, modelos generativos y [privacidad diferencial](#differential_privacy) .

### Personalización

La personalización en el contexto del aprendizaje federado es un área de investigación activa. El objetivo de la personalización es proporcionar diferentes modelos de inferencia a diferentes usuarios. Existen enfoques potencialmente diferentes para este problema.

Un enfoque es permitir que cada cliente ajuste un único modelo global (entrenado mediante aprendizaje federado) con sus datos locales. Este enfoque tiene conexiones con el metaaprendizaje; consulte, por ejemplo, [este artículo](https://arxiv.org/abs/1909.12488) . Un ejemplo de este enfoque se proporciona en [`emnist_p13n_main.py` . Para explorar y comparar diferentes estrategias de personalización, puede:](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py)

- Defina una estrategia de personalización implementando una `tf.function` parte de un modelo inicial, entrena y evalúa un modelo personalizado utilizando los conjuntos de datos locales de cada cliente. [`build_personalize_fn` proporciona un ejemplo.](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py)
- Defina un `OrderedDict` que asigne nombres de estrategias a las estrategias de personalización correspondientes y utilícelo como argumento `personalize_fn_dict` en [`tff.learning.build_personalization_eval` .](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)

Otro enfoque es evitar entrenar un modelo completamente global entrenando parte de un modelo completamente localmente. En [esta publicación de blog](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html) se describe una instancia de este enfoque. Este enfoque también está relacionado con el metaaprendizaje, consulte [este artículo](https://arxiv.org/abs/2102.03448) . Para explorar el aprendizaje federado parcialmente local, puede:

- Consulte el [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) para ver un ejemplo de código completo que aplica la reconstrucción federada y [ejercicios de seguimiento](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations) .
- Cree un proceso de capacitación parcialmente local usando [`tff.learning.reconstruction.build_training_process` , modificando `dataset_split_fn` para personalizar el comportamiento del proceso.](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process)
