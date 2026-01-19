SELECT
    "EMPRESA_SERVICIO" AS EMPRESA,
    "Año" AS ANIO, 
    "Año Periodo" AS ANIO_MES, 
    "Fecha Contable" AS FECHA,
    ( 
        SUM("Importe Boletos Taquilla") +
        SUM("Importe Boletos Intercia Ext") +
        SUM("Importe Boletos Intercia Int")
    ) AS ING_TAQ,
    ( 
        SUM("Importe Boletos A Bordo") +
        SUM("Importe Boletos Manuales") +
        SUM("Importe Boletos Portatil")
    ) AS ING_ABO,
    SUM("Importe") AS INGRESO,
    SUM("KMs Viaje") AS KMS,
    SUM("Cantidad") AS PAX,
    SUM("Cantidad Viajes") AS VIAJES,
    ( SUM("Importe") / SUM("KMs Viaje") ) AS FIK
FROM DATA
WHERE 
    ("Año" >= '2023')
GROUP BY 
    "Año",
    "Año Periodo", 
    "Fecha Contable"
ORDER BY "FECHA"