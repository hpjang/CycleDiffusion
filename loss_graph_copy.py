import matplotlib.pyplot as plt
import numpy as np

data = """
Epoch 1: loss = 0.6754, cyc_loss = 1.4059, total_loss = 2.0813
Epoch 2: loss = 0.1635, cyc_loss = 0.1267, total_loss = 0.2902
Epoch 3: loss = 0.1449, cyc_loss = 0.0717, total_loss = 0.2166
Epoch 4: loss = 0.1534, cyc_loss = 0.0530, total_loss = 0.2064
Epoch 5: loss = 0.1568, cyc_loss = 0.0418, total_loss = 0.1987
Epoch 6: loss = 0.1472, cyc_loss = 0.0461, total_loss = 0.1933
Epoch 7: loss = 0.1378, cyc_loss = 0.0459, total_loss = 0.1838
Epoch 8: loss = 0.1370, cyc_loss = 0.0463, total_loss = 0.1833
Epoch 9: loss = 0.1401, cyc_loss = 0.0418, total_loss = 0.1819
Epoch 10: loss = 0.1365, cyc_loss = 0.0379, total_loss = 0.1744
Epoch 11: loss = 0.1472, cyc_loss = 0.0314, total_loss = 0.1786
Epoch 12: loss = 0.1208, cyc_loss = 0.0476, total_loss = 0.1684
Epoch 13: loss = 0.1351, cyc_loss = 0.0430, total_loss = 0.1782
Epoch 14: loss = 0.1293, cyc_loss = 0.0416, total_loss = 0.1709
Epoch 15: loss = 0.1282, cyc_loss = 0.0565, total_loss = 0.1846
Epoch 16: loss = 0.1225, cyc_loss = 0.0341, total_loss = 0.1566
Epoch 17: loss = 0.1260, cyc_loss = 0.0385, total_loss = 0.1644
Epoch 18: loss = 0.1320, cyc_loss = 0.0435, total_loss = 0.1755
Epoch 19: loss = 0.1215, cyc_loss = 0.0474, total_loss = 0.1689
Epoch 20: loss = 0.1202, cyc_loss = 0.0474, total_loss = 0.1676
Epoch 21: loss = 0.1269, cyc_loss = 0.0385, total_loss = 0.1654
Epoch 22: loss = 0.1301, cyc_loss = 0.0575, total_loss = 0.1876
Epoch 23: loss = 0.1322, cyc_loss = 0.0372, total_loss = 0.1694
Epoch 24: loss = 0.1256, cyc_loss = 0.0423, total_loss = 0.1679
Epoch 25: loss = 0.1228, cyc_loss = 0.0373, total_loss = 0.1601
Epoch 26: loss = 0.1187, cyc_loss = 0.0469, total_loss = 0.1656
Epoch 27: loss = 0.1286, cyc_loss = 0.0472, total_loss = 0.1758
Epoch 28: loss = 0.1162, cyc_loss = 0.0445, total_loss = 0.1607
Epoch 29: loss = 0.1259, cyc_loss = 0.0480, total_loss = 0.1739
Epoch 30: loss = 0.1145, cyc_loss = 0.0538, total_loss = 0.1683
Epoch 31: loss = 0.1160, cyc_loss = 0.0383, total_loss = 0.1543
Epoch 32: loss = 0.1244, cyc_loss = 0.0335, total_loss = 0.1580
Epoch 33: loss = 0.1265, cyc_loss = 0.0459, total_loss = 0.1724
Epoch 34: loss = 0.1183, cyc_loss = 0.0382, total_loss = 0.1565
Epoch 35: loss = 0.1184, cyc_loss = 0.0419, total_loss = 0.1603
Epoch 36: loss = 0.1258, cyc_loss = 0.0348, total_loss = 0.1606
Epoch 37: loss = 0.1143, cyc_loss = 0.0444, total_loss = 0.1586
Epoch 38: loss = 0.1182, cyc_loss = 0.0380, total_loss = 0.1562
Epoch 39: loss = 0.1192, cyc_loss = 0.0378, total_loss = 0.1570
Epoch 40: loss = 0.1139, cyc_loss = 0.0431, total_loss = 0.1570
Epoch 41: loss = 0.1178, cyc_loss = 0.0414, total_loss = 0.1592
Epoch 42: loss = 0.1235, cyc_loss = 0.0507, total_loss = 0.1742
Epoch 43: loss = 0.1121, cyc_loss = 0.0401, total_loss = 0.1522
Epoch 44: loss = 0.1187, cyc_loss = 0.0362, total_loss = 0.1548
Epoch 45: loss = 0.1157, cyc_loss = 0.0338, total_loss = 0.1496
Epoch 46: loss = 0.1238, cyc_loss = 0.0407, total_loss = 0.1645
Epoch 47: loss = 0.1259, cyc_loss = 0.0526, total_loss = 0.1785
Epoch 48: loss = 0.1290, cyc_loss = 0.0375, total_loss = 0.1665
Epoch 49: loss = 0.1163, cyc_loss = 0.0409, total_loss = 0.1572
Epoch 50: loss = 0.1118, cyc_loss = 0.0372, total_loss = 0.1490
Epoch 51: loss = 0.1175, cyc_loss = 0.0511, total_loss = 0.1686
Epoch 52: loss = 0.1272, cyc_loss = 0.0398, total_loss = 0.1670
Epoch 53: loss = 0.1174, cyc_loss = 0.0353, total_loss = 0.1527
Epoch 54: loss = 0.1182, cyc_loss = 0.0391, total_loss = 0.1573
Epoch 55: loss = 0.1160, cyc_loss = 0.0464, total_loss = 0.1624
Epoch 56: loss = 0.1185, cyc_loss = 0.0372, total_loss = 0.1557
Epoch 57: loss = 0.1131, cyc_loss = 0.0342, total_loss = 0.1473
Epoch 58: loss = 0.1099, cyc_loss = 0.0361, total_loss = 0.1460
Epoch 59: loss = 0.1117, cyc_loss = 0.0493, total_loss = 0.1610
Epoch 60: loss = 0.1181, cyc_loss = 0.0402, total_loss = 0.1583
Epoch 61: loss = 0.1202, cyc_loss = 0.0486, total_loss = 0.1687
Epoch 62: loss = 0.1109, cyc_loss = 0.0551, total_loss = 0.1660
Epoch 63: loss = 0.1197, cyc_loss = 0.0332, total_loss = 0.1529
Epoch 64: loss = 0.1161, cyc_loss = 0.0592, total_loss = 0.1752
Epoch 65: loss = 0.1007, cyc_loss = 0.0332, total_loss = 0.1340
Epoch 66: loss = 0.1210, cyc_loss = 0.0284, total_loss = 0.1494
Epoch 67: loss = 0.1054, cyc_loss = 0.0333, total_loss = 0.1387
Epoch 68: loss = 0.1085, cyc_loss = 0.0374, total_loss = 0.1459
Epoch 69: loss = 0.1133, cyc_loss = 0.0373, total_loss = 0.1506
Epoch 70: loss = 0.1218, cyc_loss = 0.0364, total_loss = 0.1582
Epoch 71: loss = 0.1061, cyc_loss = 0.0390, total_loss = 0.1451
Epoch 72: loss = 0.1120, cyc_loss = 0.0385, total_loss = 0.1505
Epoch 73: loss = 0.1187, cyc_loss = 0.0438, total_loss = 0.1625
Epoch 74: loss = 0.1194, cyc_loss = 0.0432, total_loss = 0.1626
Epoch 75: loss = 0.1285, cyc_loss = 0.0336, total_loss = 0.1621
Epoch 76: loss = 0.1136, cyc_loss = 0.0386, total_loss = 0.1523
Epoch 77: loss = 0.1201, cyc_loss = 0.0421, total_loss = 0.1622
Epoch 78: loss = 0.1141, cyc_loss = 0.0323, total_loss = 0.1464
Epoch 79: loss = 0.1095, cyc_loss = 0.0355, total_loss = 0.1450
Epoch 80: loss = 0.1159, cyc_loss = 0.0475, total_loss = 0.1634
Epoch 81: loss = 0.1154, cyc_loss = 0.0343, total_loss = 0.1497
Epoch 82: loss = 0.1103, cyc_loss = 0.0383, total_loss = 0.1486
Epoch 83: loss = 0.1183, cyc_loss = 0.0371, total_loss = 0.1554
Epoch 84: loss = 0.1202, cyc_loss = 0.0403, total_loss = 0.1605
Epoch 85: loss = 0.1110, cyc_loss = 0.0379, total_loss = 0.1489
Epoch 86: loss = 0.1243, cyc_loss = 0.0301, total_loss = 0.1544
Epoch 87: loss = 0.1176, cyc_loss = 0.0318, total_loss = 0.1494
Epoch 88: loss = 0.1101, cyc_loss = 0.0352, total_loss = 0.1453
Epoch 89: loss = 0.1119, cyc_loss = 0.0361, total_loss = 0.1480
Epoch 90: loss = 0.1195, cyc_loss = 0.0311, total_loss = 0.1506
Epoch 91: loss = 0.1119, cyc_loss = 0.0331, total_loss = 0.1450
Epoch 92: loss = 0.1111, cyc_loss = 0.0359, total_loss = 0.1470
Epoch 93: loss = 0.1052, cyc_loss = 0.0337, total_loss = 0.1390
Epoch 94: loss = 0.1155, cyc_loss = 0.0334, total_loss = 0.1489
Epoch 95: loss = 0.1162, cyc_loss = 0.0384, total_loss = 0.1545
Epoch 96: loss = 0.1113, cyc_loss = 0.0446, total_loss = 0.1559
Epoch 97: loss = 0.1221, cyc_loss = 0.0333, total_loss = 0.1553
Epoch 98: loss = 0.1112, cyc_loss = 0.0425, total_loss = 0.1537
Epoch 99: loss = 0.1028, cyc_loss = 0.0336, total_loss = 0.1364
Epoch 100: loss = 0.1077, cyc_loss = 0.0480, total_loss = 0.1557
Epoch 101: loss = 0.1185, cyc_loss = 0.0348, total_loss = 0.1533
Epoch 102: loss = 0.1105, cyc_loss = 0.0300, total_loss = 0.1406
Epoch 103: loss = 0.1128, cyc_loss = 0.0291, total_loss = 0.1418
Epoch 104: loss = 0.1070, cyc_loss = 0.0286, total_loss = 0.1356
Epoch 105: loss = 0.1157, cyc_loss = 0.0327, total_loss = 0.1484
Epoch 106: loss = 0.1107, cyc_loss = 0.0385, total_loss = 0.1492
Epoch 107: loss = 0.1190, cyc_loss = 0.0317, total_loss = 0.1507
Epoch 108: loss = 0.1145, cyc_loss = 0.0444, total_loss = 0.1589
Epoch 109: loss = 0.1187, cyc_loss = 0.0350, total_loss = 0.1537
Epoch 110: loss = 0.1151, cyc_loss = 0.0453, total_loss = 0.1604
Epoch 111: loss = 0.1165, cyc_loss = 0.0391, total_loss = 0.1556
Epoch 112: loss = 0.1238, cyc_loss = 0.0367, total_loss = 0.1605
Epoch 113: loss = 0.1184, cyc_loss = 0.0405, total_loss = 0.1590
Epoch 114: loss = 0.1175, cyc_loss = 0.0304, total_loss = 0.1479
Epoch 115: loss = 0.1091, cyc_loss = 0.0352, total_loss = 0.1443
Epoch 116: loss = 0.1309, cyc_loss = 0.0368, total_loss = 0.1676
Epoch 117: loss = 0.1148, cyc_loss = 0.0405, total_loss = 0.1553
Epoch 118: loss = 0.1187, cyc_loss = 0.0399, total_loss = 0.1586
Epoch 119: loss = 0.1092, cyc_loss = 0.0528, total_loss = 0.1620
Epoch 120: loss = 0.1226, cyc_loss = 0.0437, total_loss = 0.1663
Epoch 121: loss = 0.1082, cyc_loss = 0.0346, total_loss = 0.1428
Epoch 122: loss = 0.1062, cyc_loss = 0.0335, total_loss = 0.1397
Epoch 123: loss = 0.1053, cyc_loss = 0.0410, total_loss = 0.1464
Epoch 124: loss = 0.1102, cyc_loss = 0.0309, total_loss = 0.1410
Epoch 125: loss = 0.1148, cyc_loss = 0.0350, total_loss = 0.1498
Epoch 126: loss = 0.1176, cyc_loss = 0.0333, total_loss = 0.1509
Epoch 127: loss = 0.1063, cyc_loss = 0.0309, total_loss = 0.1372
Epoch 128: loss = 0.1098, cyc_loss = 0.0298, total_loss = 0.1396
Epoch 129: loss = 0.1078, cyc_loss = 0.0380, total_loss = 0.1459
Epoch 130: loss = 0.1178, cyc_loss = 0.0292, total_loss = 0.1470
Epoch 131: loss = 0.1137, cyc_loss = 0.0395, total_loss = 0.1532
Epoch 132: loss = 0.1038, cyc_loss = 0.0335, total_loss = 0.1373
Epoch 133: loss = 0.1169, cyc_loss = 0.0325, total_loss = 0.1494
Epoch 134: loss = 0.1060, cyc_loss = 0.0314, total_loss = 0.1374
Epoch 135: loss = 0.1126, cyc_loss = 0.0352, total_loss = 0.1477
Epoch 136: loss = 0.1125, cyc_loss = 0.0409, total_loss = 0.1534
Epoch 137: loss = 0.1150, cyc_loss = 0.0423, total_loss = 0.1573
Epoch 138: loss = 0.1059, cyc_loss = 0.0278, total_loss = 0.1337
Epoch 139: loss = 0.1120, cyc_loss = 0.0299, total_loss = 0.1419
Epoch 140: loss = 0.1037, cyc_loss = 0.0273, total_loss = 0.1311
Epoch 141: loss = 0.1152, cyc_loss = 0.0396, total_loss = 0.1548
Epoch 142: loss = 0.1153, cyc_loss = 0.0352, total_loss = 0.1505
Epoch 143: loss = 0.0963, cyc_loss = 0.0337, total_loss = 0.1301
Epoch 144: loss = 0.1110, cyc_loss = 0.0290, total_loss = 0.1400
Epoch 145: loss = 0.1141, cyc_loss = 0.0330, total_loss = 0.1471
Epoch 146: loss = 0.1109, cyc_loss = 0.0459, total_loss = 0.1568
Epoch 147: loss = 0.1124, cyc_loss = 0.0268, total_loss = 0.1392
Epoch 148: loss = 0.1198, cyc_loss = 0.0370, total_loss = 0.1568
Epoch 149: loss = 0.1052, cyc_loss = 0.0325, total_loss = 0.1377
Epoch 150: loss = 0.1079, cyc_loss = 0.0289, total_loss = 0.1368
Epoch 151: loss = 0.1131, cyc_loss = 0.0338, total_loss = 0.1469
Epoch 152: loss = 0.1103, cyc_loss = 0.0326, total_loss = 0.1428
Epoch 153: loss = 0.1111, cyc_loss = 0.0318, total_loss = 0.1430
Epoch 154: loss = 0.1188, cyc_loss = 0.0345, total_loss = 0.1533
Epoch 155: loss = 0.1088, cyc_loss = 0.0302, total_loss = 0.1390
Epoch 156: loss = 0.1169, cyc_loss = 0.0333, total_loss = 0.1502
Epoch 157: loss = 0.1102, cyc_loss = 0.0269, total_loss = 0.1371
Epoch 158: loss = 0.1033, cyc_loss = 0.0290, total_loss = 0.1323
Epoch 159: loss = 0.1049, cyc_loss = 0.0302, total_loss = 0.1351
Epoch 160: loss = 0.1195, cyc_loss = 0.0303, total_loss = 0.1497
Epoch 161: loss = 0.1100, cyc_loss = 0.0350, total_loss = 0.1450
Epoch 162: loss = 0.1047, cyc_loss = 0.0417, total_loss = 0.1463
Epoch 163: loss = 0.1038, cyc_loss = 0.0359, total_loss = 0.1398
Epoch 164: loss = 0.1059, cyc_loss = 0.0332, total_loss = 0.1391
Epoch 165: loss = 0.1074, cyc_loss = 0.0296, total_loss = 0.1370
Epoch 166: loss = 0.1101, cyc_loss = 0.0348, total_loss = 0.1449
Epoch 167: loss = 0.1142, cyc_loss = 0.0330, total_loss = 0.1471
Epoch 168: loss = 0.1132, cyc_loss = 0.0320, total_loss = 0.1452
Epoch 169: loss = 0.1168, cyc_loss = 0.0294, total_loss = 0.1461
Epoch 170: loss = 0.1141, cyc_loss = 0.0279, total_loss = 0.1421
Epoch 171: loss = 0.1043, cyc_loss = 0.0402, total_loss = 0.1446
Epoch 172: loss = 0.1188, cyc_loss = 0.0277, total_loss = 0.1464
Epoch 173: loss = 0.1085, cyc_loss = 0.0292, total_loss = 0.1376
Epoch 174: loss = 0.1172, cyc_loss = 0.0310, total_loss = 0.1482
Epoch 175: loss = 0.1086, cyc_loss = 0.0288, total_loss = 0.1374
Epoch 176: loss = 0.1211, cyc_loss = 0.0324, total_loss = 0.1536
Epoch 177: loss = 0.1064, cyc_loss = 0.0315, total_loss = 0.1379
Epoch 178: loss = 0.1101, cyc_loss = 0.0416, total_loss = 0.1518
Epoch 179: loss = 0.1045, cyc_loss = 0.0353, total_loss = 0.1398
Epoch 180: loss = 0.1123, cyc_loss = 0.0259, total_loss = 0.1382
Epoch 181: loss = 0.1127, cyc_loss = 0.0343, total_loss = 0.1470
Epoch 182: loss = 0.1089, cyc_loss = 0.0333, total_loss = 0.1422
Epoch 183: loss = 0.1191, cyc_loss = 0.0485, total_loss = 0.1677
Epoch 184: loss = 0.1065, cyc_loss = 0.0338, total_loss = 0.1403
Epoch 185: loss = 0.1093, cyc_loss = 0.0347, total_loss = 0.1439
Epoch 186: loss = 0.1186, cyc_loss = 0.0278, total_loss = 0.1464
Epoch 187: loss = 0.0952, cyc_loss = 0.0275, total_loss = 0.1227
Epoch 188: loss = 0.1093, cyc_loss = 0.0386, total_loss = 0.1478
Epoch 189: loss = 0.1090, cyc_loss = 0.0298, total_loss = 0.1388
Epoch 190: loss = 0.1122, cyc_loss = 0.0303, total_loss = 0.1425
Epoch 191: loss = 0.1059, cyc_loss = 0.0304, total_loss = 0.1362
Epoch 192: loss = 0.1074, cyc_loss = 0.0304, total_loss = 0.1378
Epoch 193: loss = 0.1120, cyc_loss = 0.0321, total_loss = 0.1441
Epoch 194: loss = 0.1075, cyc_loss = 0.0345, total_loss = 0.1419
Epoch 195: loss = 0.1055, cyc_loss = 0.0389, total_loss = 0.1444
Epoch 196: loss = 0.1144, cyc_loss = 0.0307, total_loss = 0.1451
Epoch 197: loss = 0.1155, cyc_loss = 0.0274, total_loss = 0.1428
Epoch 198: loss = 0.1058, cyc_loss = 0.0266, total_loss = 0.1324
Epoch 199: loss = 0.1036, cyc_loss = 0.0311, total_loss = 0.1347
Epoch 200: loss = 0.1100, cyc_loss = 0.0353, total_loss = 0.1453
Epoch 201: loss = 0.1042, cyc_loss = 0.0291, total_loss = 0.1333
Epoch 202: loss = 0.1191, cyc_loss = 0.0262, total_loss = 0.1453
Epoch 203: loss = 0.1124, cyc_loss = 0.0316, total_loss = 0.1440
Epoch 204: loss = 0.1124, cyc_loss = 0.0311, total_loss = 0.1435
Epoch 205: loss = 0.1041, cyc_loss = 0.0360, total_loss = 0.1401
Epoch 206: loss = 0.1106, cyc_loss = 0.0349, total_loss = 0.1455
Epoch 207: loss = 0.1198, cyc_loss = 0.0353, total_loss = 0.1551
Epoch 208: loss = 0.0991, cyc_loss = 0.0256, total_loss = 0.1248
Epoch 209: loss = 0.1075, cyc_loss = 0.0262, total_loss = 0.1337
Epoch 210: loss = 0.1033, cyc_loss = 0.0288, total_loss = 0.1321
Epoch 211: loss = 0.1017, cyc_loss = 0.0267, total_loss = 0.1283
Epoch 212: loss = 0.1084, cyc_loss = 0.0264, total_loss = 0.1349
Epoch 213: loss = 0.1099, cyc_loss = 0.0364, total_loss = 0.1463
Epoch 214: loss = 0.1089, cyc_loss = 0.0322, total_loss = 0.1411
Epoch 215: loss = 0.1094, cyc_loss = 0.0325, total_loss = 0.1418
Epoch 216: loss = 0.1070, cyc_loss = 0.0251, total_loss = 0.1321
Epoch 217: loss = 0.1001, cyc_loss = 0.0289, total_loss = 0.1291
Epoch 218: loss = 0.1104, cyc_loss = 0.0364, total_loss = 0.1468
Epoch 219: loss = 0.0944, cyc_loss = 0.0345, total_loss = 0.1289
Epoch 220: loss = 0.1024, cyc_loss = 0.0362, total_loss = 0.1385
Epoch 221: loss = 0.1092, cyc_loss = 0.0325, total_loss = 0.1417
Epoch 222: loss = 0.1003, cyc_loss = 0.0333, total_loss = 0.1337
Epoch 223: loss = 0.1135, cyc_loss = 0.0342, total_loss = 0.1477
Epoch 224: loss = 0.1072, cyc_loss = 0.0307, total_loss = 0.1379
Epoch 225: loss = 0.1109, cyc_loss = 0.0330, total_loss = 0.1439
Epoch 226: loss = 0.1012, cyc_loss = 0.0316, total_loss = 0.1328
Epoch 227: loss = 0.1104, cyc_loss = 0.0342, total_loss = 0.1445
Epoch 228: loss = 0.1118, cyc_loss = 0.0362, total_loss = 0.1481
Epoch 229: loss = 0.0992, cyc_loss = 0.0392, total_loss = 0.1384
Epoch 230: loss = 0.1072, cyc_loss = 0.0285, total_loss = 0.1356
Epoch 231: loss = 0.0895, cyc_loss = 0.0301, total_loss = 0.1196
Epoch 232: loss = 0.1052, cyc_loss = 0.0295, total_loss = 0.1347
Epoch 233: loss = 0.1019, cyc_loss = 0.0299, total_loss = 0.1318
Epoch 234: loss = 0.1039, cyc_loss = 0.0316, total_loss = 0.1354
Epoch 235: loss = 0.1094, cyc_loss = 0.0263, total_loss = 0.1358
Epoch 236: loss = 0.1066, cyc_loss = 0.0283, total_loss = 0.1349
Epoch 237: loss = 0.1027, cyc_loss = 0.0356, total_loss = 0.1383
Epoch 238: loss = 0.1178, cyc_loss = 0.0309, total_loss = 0.1487
Epoch 239: loss = 0.1120, cyc_loss = 0.0251, total_loss = 0.1371
Epoch 240: loss = 0.1054, cyc_loss = 0.0326, total_loss = 0.1379
Epoch 241: loss = 0.1065, cyc_loss = 0.0311, total_loss = 0.1376
Epoch 242: loss = 0.1060, cyc_loss = 0.0345, total_loss = 0.1405
Epoch 243: loss = 0.1129, cyc_loss = 0.0306, total_loss = 0.1435
Epoch 244: loss = 0.1164, cyc_loss = 0.0443, total_loss = 0.1606
Epoch 245: loss = 0.0969, cyc_loss = 0.0332, total_loss = 0.1301
Epoch 246: loss = 0.1069, cyc_loss = 0.0317, total_loss = 0.1386
Epoch 247: loss = 0.0966, cyc_loss = 0.0384, total_loss = 0.1350
Epoch 248: loss = 0.0973, cyc_loss = 0.0342, total_loss = 0.1315
Epoch 249: loss = 0.1051, cyc_loss = 0.0310, total_loss = 0.1361
Epoch 250: loss = 0.1149, cyc_loss = 0.0283, total_loss = 0.1432
Epoch 251: loss = 0.1106, cyc_loss = 0.0236, total_loss = 0.1342
Epoch 252: loss = 0.1143, cyc_loss = 0.0330, total_loss = 0.1473
Epoch 253: loss = 0.0986, cyc_loss = 0.0324, total_loss = 0.1311
Epoch 254: loss = 0.1043, cyc_loss = 0.0320, total_loss = 0.1363
Epoch 255: loss = 0.1104, cyc_loss = 0.0272, total_loss = 0.1376
Epoch 256: loss = 0.1050, cyc_loss = 0.0285, total_loss = 0.1335
Epoch 257: loss = 0.1015, cyc_loss = 0.0304, total_loss = 0.1319
Epoch 258: loss = 0.1007, cyc_loss = 0.0313, total_loss = 0.1321
Epoch 259: loss = 0.1088, cyc_loss = 0.0350, total_loss = 0.1437
Epoch 260: loss = 0.1000, cyc_loss = 0.0346, total_loss = 0.1346
Epoch 261: loss = 0.1035, cyc_loss = 0.0278, total_loss = 0.1312
Epoch 262: loss = 0.1035, cyc_loss = 0.0356, total_loss = 0.1391
Epoch 263: loss = 0.1064, cyc_loss = 0.0337, total_loss = 0.1401
Epoch 264: loss = 0.0959, cyc_loss = 0.0291, total_loss = 0.1251
Epoch 265: loss = 0.0991, cyc_loss = 0.0278, total_loss = 0.1268
Epoch 266: loss = 0.1132, cyc_loss = 0.0250, total_loss = 0.1382
Epoch 267: loss = 0.1042, cyc_loss = 0.0288, total_loss = 0.1330
Epoch 268: loss = 0.0990, cyc_loss = 0.0295, total_loss = 0.1286
Epoch 269: loss = 0.1151, cyc_loss = 0.0331, total_loss = 0.1482
Epoch 270: loss = 0.1115, cyc_loss = 0.0277, total_loss = 0.1392
Epoch 271: loss = 0.1102, cyc_loss = 0.0294, total_loss = 0.1396
Epoch 272: loss = 0.1009, cyc_loss = 0.0287, total_loss = 0.1295
Epoch 273: loss = 0.1044, cyc_loss = 0.0301, total_loss = 0.1345
Epoch 274: loss = 0.1016, cyc_loss = 0.0259, total_loss = 0.1275
Epoch 275: loss = 0.0982, cyc_loss = 0.0296, total_loss = 0.1278
Epoch 276: loss = 0.0968, cyc_loss = 0.0284, total_loss = 0.1252
Epoch 277: loss = 0.1005, cyc_loss = 0.0298, total_loss = 0.1303
Epoch 278: loss = 0.1076, cyc_loss = 0.0266, total_loss = 0.1342
Epoch 279: loss = 0.0968, cyc_loss = 0.0289, total_loss = 0.1258
Epoch 280: loss = 0.0977, cyc_loss = 0.0271, total_loss = 0.1248
Epoch 281: loss = 0.0961, cyc_loss = 0.0279, total_loss = 0.1240
Epoch 282: loss = 0.1007, cyc_loss = 0.0281, total_loss = 0.1288
Epoch 283: loss = 0.0992, cyc_loss = 0.0295, total_loss = 0.1286
Epoch 284: loss = 0.0968, cyc_loss = 0.0339, total_loss = 0.1307
Epoch 285: loss = 0.1071, cyc_loss = 0.0333, total_loss = 0.1404
Epoch 286: loss = 0.1008, cyc_loss = 0.0242, total_loss = 0.1250
Epoch 287: loss = 0.0975, cyc_loss = 0.0298, total_loss = 0.1273
Epoch 288: loss = 0.1073, cyc_loss = 0.0302, total_loss = 0.1375
Epoch 289: loss = 0.1004, cyc_loss = 0.0309, total_loss = 0.1314
Epoch 290: loss = 0.1010, cyc_loss = 0.0359, total_loss = 0.1369
Epoch 291: loss = 0.1021, cyc_loss = 0.0266, total_loss = 0.1288
Epoch 292: loss = 0.1079, cyc_loss = 0.0356, total_loss = 0.1435
Epoch 293: loss = 0.0982, cyc_loss = 0.0283, total_loss = 0.1265
Epoch 294: loss = 0.0998, cyc_loss = 0.0245, total_loss = 0.1243
Epoch 295: loss = 0.1064, cyc_loss = 0.0307, total_loss = 0.1371
Epoch 296: loss = 0.1027, cyc_loss = 0.0240, total_loss = 0.1268
Epoch 297: loss = 0.0995, cyc_loss = 0.0244, total_loss = 0.1239
Epoch 298: loss = 0.1081, cyc_loss = 0.0238, total_loss = 0.1319
Epoch 299: loss = 0.1015, cyc_loss = 0.0288, total_loss = 0.1302
Epoch 300: loss = 0.0991, cyc_loss = 0.0379, total_loss = 0.1370


"""

# 각 줄을 분할하여 필요한 값을 추출하고 리스트에 저장
loss_list = [float(line.split(",")[0].split("=")[1]) for line in data.split("\n") if line.strip()]
cyc_loss_list = [float(line.split(",")[1].split("=")[1]) for line in data.split("\n") if line.strip()]
total_loss_list = [float(line.split(",")[2].split("=")[1]) for line in data.split("\n") if line.strip()]

# Epoch 번호 (1부터 시작)
epochs = range(1, len(loss_list) + 1)


# y축 범위 설정
plt.ylim(0.0, 0.3)
# y축 틱 설정
plt.yticks(np.arange(0.0, 0.3, 0.1))


# Loss 그래프
plt.plot(epochs, loss_list, label='Loss', color='blue')
# Cycle Loss 그래프
plt.plot(epochs, cyc_loss_list, label='Cycle Loss', color='red')
# Total Loss 그래프
plt.plot(epochs, total_loss_list, label='Total Loss', color='green')

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# 이미지로 저장
plt.savefig('test_tgt1_time6.png')
