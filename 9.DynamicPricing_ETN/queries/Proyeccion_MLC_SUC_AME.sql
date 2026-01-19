SELECT
    'AME' as EMPRESA,
    "Año" AS ANIO, 
    "Año Periodo" AS ANIO_MES, 
    "Fecha Contable" AS FECHA,
    SUM("Importe") AS INGRESO, 
    SUM("KMs Viaje") AS KMS,
    SUM("CANTIDAD") AS PAX
FROM DATA
WHERE 
    ("Año" >= '2023')  
GROUP BY 
    "Año",
    "Año Periodo", 
    "Fecha Contable"
ORDER BY "FECHA"