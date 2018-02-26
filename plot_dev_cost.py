import matplotlib.pyplot as plt
import numpy as np



nonpos_new_test_loss = [4941.971234975577, 4845.882110728386, 4243.276322041367, 3862.243792048137, 3628.7316402065944, 3389.2017448786314, 3279.487311872971, 3190.6096035015034, 3110.963552774352, 2984.5599260355375, 2926.234519686494, 2857.412090450509, 2818.6852614289332, 2798.47735478401, 2739.7266542054267, 2759.608200798637, 2728.3707591097445, 2687.375503676838, 2701.7678746148513, 2690.2786880426065, 2653.47418380211, 2646.6957418801085, 2628.653210838043, 2643.9441591413824, 2608.1026855889363, 2647.8457684302884, 2664.829745959202, 2662.296554576726, 2609.6390895548043, 2617.3389736743693, 2599.4120251137824, 2612.7875773043042, 2622.7206940878527, 2594.8807152042227, 2518.6185605666165, 2606.4357696102315, 2501.1565124341582, 2589.5740151890295, 2602.0034273544284, 2538.6314116640583, 2500.5004891331646, 2531.8633953579983, 2493.1544182485227, 2527.2226344707424, 2505.5313137762714, 2555.16129134186, 2478.7044380004554, 2590.131038616748, 2539.5898921010307, 2528.2435028543787, 2518.1364456963106, 2555.3865641499124, 2649.385625324272, 2547.666704884216, 2486.873644498818, 2476.144080188058, 2563.307790908253, 2472.3746734910073, 2533.5353457941637, 2549.49914772589, 2583.4756909030343, 2620.2488597656147, 2506.736618752603, 2545.364180132658, 2510.1669016761816, 2545.5800190481377, 2537.977604271349, 2578.616022401367, 2425.7158537349587, 2521.9938433369894, 2444.3894823894834, 2543.3170289817417, 2565.4812061789776, 2554.8811609071013, 2567.430275986828, 2562.5631410453925, 2557.2275743294595, 2609.3969666217426, 2513.5346294968595, 2620.2119095108355, 2528.8570990843373, 2480.0291023948444, 2529.454656103002, 2491.4285279100523, 2435.4505317399316, 2592.0967657501105, 2661.729451350131, 2550.1347536821777, 2503.9658326671374, 2575.7246743917053, 2564.2202612052947, 2559.313098502239, 2540.4537480977824, 2656.227173248656, 2558.448862345962, 2623.1484219483027, 2570.089037836945, 2570.5724243088657, 2522.0018185687686, 2522.7594938474886]
nonpos_new_test_acc = [0.4742492870323771, 0.5651736285858078, 0.6555946988760275, 0.692836772353632, 0.7116255661801711, 0.7486998825700386, 0.7513839959738299, 0.7594363361852038, 0.7663143767824191, 0.7829223284683778, 0.7884583123636973, 0.7953363529609127, 0.8007045797684952, 0.8002013085052844, 0.8025499077336018, 0.8023821506458648, 0.8042274786109713, 0.813286361348767, 0.8070793491024996, 0.8089246770676061, 0.8142929038751887, 0.813454118436504, 0.8142929038751887, 0.8168092601912431, 0.8213387015601409, 0.817144774366717, 0.8114410333836605, 0.8102667337695018, 0.8141251467874517, 0.8201644019459822, 0.8188223452440866, 0.8183190739808757, 0.8188223452440866, 0.8204999161214561, 0.8278812279818822, 0.8154672034893474, 0.8278812279818822, 0.817144774366717, 0.8173125314544539, 0.8211709444724039, 0.8262036571045127, 0.8221774869988256, 0.8265391712799866, 0.8216742157356148, 0.8260359000167757, 0.8193256165072974, 0.8257003858413018, 0.8174802885421909, 0.8225130011742996, 0.8255326287535648, 0.8236873007884583, 0.8215064586478779, 0.8116087904713974, 0.8196611306827714, 0.829055527596041, 0.824861600402617, 0.8228485153497735, 0.8280489850696192, 0.8216742157356148, 0.825029357490354, 0.8188223452440866, 0.8161382318402952, 0.828384499245093, 0.8231840295252475, 0.824861600402617, 0.8223452440865626, 0.825197114578091, 0.8226807582620366, 0.8337527260526757, 0.8297265559469887, 0.832578426438517, 0.8208354302969301, 0.8241905720516692, 0.8221774869988256, 0.8201644019459822, 0.8199966448582453, 0.8204999161214561, 0.8262036571045127, 0.8272101996309345, 0.8191578594195604, 0.8270424425431975, 0.8320751551753062, 0.8236873007884583, 0.8295587988592518, 0.8339204831404127, 0.8208354302969301, 0.8196611306827714, 0.8216742157356148, 0.832746183526254, 0.8268746854554605, 0.8260359000167757, 0.8240228149639323, 0.8309008555611475, 0.8184868310686126, 0.8273779567186713, 0.8194933735950344, 0.82855225633283, 0.8309008555611475, 0.8292232846837779, 0.8320751551753062]
nonpos_dev_acc =  [0.8700319375443577, 0.9054293825408091, 0.9279630943931867, 0.9378105039034776, 0.9441092973740242, 0.9486337828246983, 0.9504080908445706, 0.9520049680624556, 0.9533356990773598, 0.9564407381121363, 0.9567955997161107, 0.9581263307310149, 0.9584811923349894, 0.9592796309439319, 0.9605216465578424, 0.9614088005677786, 0.9614088005677786, 0.962473385379702, 0.9631831085876508, 0.9635379701916252, 0.9636266855926189, 0.964070262597587, 0.9643364088005678, 0.9646025550035486, 0.9648687012065295, 0.9653122782114976, 0.9662881476224272, 0.9667317246273953, 0.9676188786373314, 0.96611071682044, 0.9675301632363378, 0.9669091554293825, 0.966820440028389, 0.9675301632363378, 0.9675301632363378, 0.9681511710432931, 0.9677075940383251, 0.9686834634492548, 0.9685947480482612, 0.9693044712562101, 0.9689496096522356, 0.9691270404542228, 0.9693044712562101, 0.9702803406671399, 0.9695706174591909, 0.9703690560681334, 0.970457771469127, 0.9702803406671399, 0.9708126330731015, 0.9700141944641589, 0.9712562100780695, 0.9713449254790631, 0.970457771469127, 0.971611071682044, 0.9710787792760823, 0.9720546486870121, 0.9720546486870121, 0.972764371894961, 0.9729418026969482, 0.972764371894961, 0.9732966643009227, 0.9736515259048971, 0.9724095102909865, 0.9736515259048971, 0.9740063875088716, 0.9740063875088716, 0.9736515259048971, 0.973207948899929, 0.9745386799148332, 0.9735628105039035, 0.9742725337118524, 0.9743612491128459, 0.9740951029098651, 0.9737402413058907, 0.9750709723207949, 0.9742725337118524, 0.9746273953158269, 0.9739176721078779, 0.9750709723207949, 0.9750709723207949, 0.9748935415188077, 0.9749822569198012, 0.9756032647267565, 0.9754258339247693, 0.9754258339247693, 0.9752484031227822, 0.9760468417317246, 0.9748935415188077, 0.9762242725337118, 0.9761355571327183, 0.975958126330731, 0.9760468417317246, 0.9762242725337118, 0.975958126330731, 0.9763129879347054, 0.9749822569198012, 0.977111426543648, 0.9753371185237757, 0.9768452803406671, 0.9769339957416607]
nonpos_dev_loss = [3945.482533254747, 2814.1689506369903, 2233.9177542548987, 1902.4806636883718, 1713.9537834792618, 1602.1971209413343, 1505.3630558153577, 1451.1280742193999, 1409.6616022560088, 1364.2031086900797, 1332.958485260799, 1303.6582497099773, 1288.5306052047272, 1260.4796334221703, 1242.162530717942, 1226.7770086177939, 1215.051615101496, 1197.5068600576246, 1183.6127210306454, 1167.6860425452874, 1152.1111114171704, 1140.0003147758455, 1132.381099910127, 1118.7778936776199, 1108.9256500115193, 1097.8108600643118, 1093.2330266474123, 1090.8168845611158, 1075.9707834491837, 1062.027035323763, 1054.0252029230653, 1042.5193940659788, 1039.0384285837383, 1026.5174846993498, 1019.1323527561601, 1014.5790048177026, 1005.950789779363, 999.2098337890651, 987.983484568361, 981.9229714273072, 973.4033596397078, 967.9906583643005, 960.8084340598874, 953.4586959129751, 945.616021919378, 952.5492527329324, 932.2792090149094, 927.4799500145467, 919.3101661438205, 927.3125634867072, 910.9573979243994, 902.2946083240053, 929.9996920066684, 899.3897624374831, 893.0616232208955, 885.0132080142282, 882.664827974112, 873.8208265511535, 872.9225020902793, 863.1794044249782, 863.5533281372782, 861.2013901703497, 855.513301798168, 849.2022990559681, 845.7337967216644, 839.4524132205565, 837.3889375535082, 834.7608780189669, 828.1187405195416, 833.7400564967153, 824.6114841351175, 820.9517727167081, 818.711006833404, 822.255682312682, 819.9799501047657, 818.5882232921831, 808.9568055485313, 816.233535434516, 799.2831561985777, 799.418613163159, 794.1800201186479, 791.9339611386291, 793.6348495944019, 786.3682622321687, 787.9109300205192, 790.0360590905304, 783.1549231053696, 785.8165437151081, 776.6429778248195, 774.1356264849867, 772.6252905967012, 770.7334654011819, 775.6135139747462, 768.4015973388906, 768.419166629368, 778.9363689163108, 759.406531718615, 773.9450914832007, 762.2161258020881, 756.9387595689374]
pos_new_test_acc = [0.5560590802282646, 0.5683115139308493, 0.590298757972474, 0.622860020140987, 0.648539778449144, 0.6663309835515273, 0.6866398120174555, 0.7030882846592816, 0.7218865391070829, 0.7297750923128566, 0.7418596844578718, 0.7527693856998993, 0.7621685129238, 0.7707284323598523, 0.775260154414233, 0.7809667673716012, 0.7863376972138302, 0.7910372608257804, 0.7987579724739846, 0.8004363880496811, 0.8016112789526687, 0.8088284659281638, 0.8089963074857335, 0.8128566633098355, 0.8202416918429003, 0.8165491775763679, 0.8187311178247734, 0.8244377307821417, 0.8259483048002686, 0.8229271567640147, 0.824269889224572, 0.8244377307821417, 0.8306478684122188, 0.8299765021819403, 0.8345082242363209, 0.8368580060422961, 0.8366901644847264, 0.8345082242363209, 0.8383685800604229, 0.8388721047331319, 0.8393756294058409, 0.8392077878482712, 0.8371936891574354, 0.8366901644847264, 0.8444108761329305, 0.8454179254783485, 0.8457536085934877, 0.8462571332661967, 0.8392077878482712, 0.8499496475327291, 0.8398791540785498, 0.850453172205438, 0.8499496475327291, 0.8538100033568311, 0.850956696878147, 0.8481033903994629, 0.8475998657267539, 0.8486069150721719, 0.8475998657267539, 0.8481033903994629, 0.846592816381336, 0.8573346760657939, 0.8497818059751594, 0.8524672708962739, 0.8481033903994629, 0.8613628734474655, 0.8561597851628063, 0.8469284994964753, 0.8616985565626049, 0.8439073514602216, 0.856327626720376, 0.850956696878147, 0.8593487747566297, 0.861195031889896, 0.855320577374958, 0.8596844578717691, 0.8499496475327291, 0.8541456864719704, 0.8541456864719704, 0.8637126552534408, 0.8623699227928835, 0.8615307150050352, 0.8625377643504532, 0.8608593487747567, 0.8630412890231621, 0.8586774085263511, 0.8638804968110104, 0.8645518630412891, 0.860187982544478, 0.856831151393085, 0.8653910708291372, 0.8581738838536421, 0.8610271903323263, 0.8658945955018462, 0.8669016448472642, 0.8620342396777442, 0.8608593487747567, 0.8603558241020477, 0.8605236656596174, 0.8645518630412891]
pos_new_test_loss = [4247.234530534683, 4528.144085764059, 4587.819623014701, 4453.561973958174, 4291.272589724311, 4103.449347340949, 3915.83417238956, 3731.4699061837214, 3573.726880571122, 3484.245147749729, 3336.7019171893517, 3236.8493188910666, 3146.380323458072, 3051.857794773494, 2998.935065560062, 2919.2739792939246, 2888.0181239641934, 2842.469095074812, 2765.1310090104494, 2740.0181227524263, 2728.1585567680645, 2638.0963965157835, 2632.76388058859, 2598.791381156293, 2553.245060701708, 2549.850012077907, 2533.4288201164777, 2494.263046782139, 2476.314431749128, 2473.0621181443007, 2454.8139820108727, 2445.937147153774, 2405.8096989185306, 2395.413563492272, 2375.9472969192257, 2351.161321465229, 2352.561272512046, 2356.5070977795967, 2327.8455883343577, 2321.1990843439125, 2300.164127906386, 2344.170471872193, 2355.6899957043292, 2375.6833238387044, 2291.3214122002696, 2264.4960061131505, 2269.4084778694523, 2270.6586524790036, 2332.506202296854, 2238.6970920439326, 2323.134063318882, 2220.3635329474646, 2240.8847558061375, 2187.0657452572527, 2216.8177553034084, 2247.8045228801784, 2236.748425452773, 2224.5228598686144, 2236.9320836872384, 2227.171793074653, 2257.7459598408223, 2160.0794629488437, 2221.953768558208, 2199.98115948988, 2217.6834505427164, 2127.993890193136, 2164.1930466793865, 2255.7925731678374, 2136.134829867206, 2271.3286879567268, 2186.5700666823354, 2216.845275910001, 2157.6161697391262, 2151.6263413595725, 2183.4084136313736, 2139.1205761196134, 2223.0899312444412, 2195.2165826031837, 2199.774994921082, 2119.1839283187833, 2135.8536284482175, 2133.9480131923474, 2119.4001146397895, 2116.0322630920455, 2106.8600436318457, 2150.4803245637386, 2096.3423321890805, 2082.941798785496, 2117.3877599628267, 2154.7927364471298, 2080.3689075019247, 2144.838577308871, 2108.084968460645, 2069.811114479523, 2098.2035632869606, 2120.034038990167, 2100.769352035965, 2129.8960498243187, 2121.57110891105, 2093.0024301662347]
pos_dev_acc = [0.8304633410260962, 0.8565595597372626, 0.8717379726611042, 0.8807030001775253, 0.8890466891532043, 0.8957926504526895, 0.9010296467246582, 0.9046689153204331, 0.9078643706728209, 0.9103497248357891, 0.9124800284040476, 0.9139002307828865, 0.9151429078643707, 0.9162968222971773, 0.9170069234865968, 0.9193147523522102, 0.9212675306231138, 0.9224214450559205, 0.923752884786082, 0.9249955618675662, 0.92659328954376, 0.9267708148411149, 0.9278359666252441, 0.929256169004083, 0.9288123557606959, 0.930676371382922, 0.9318302858157287, 0.9309426593289544, 0.9328954375998579, 0.9333392508432452, 0.933694301437955, 0.9334280134919226, 0.9350257411681164, 0.934936978519439, 0.9360908929522457, 0.9361796556009231, 0.9360908929522457, 0.9367122314929878, 0.9369785194390201, 0.9372448073850523, 0.937777383277117, 0.9370672820876975, 0.9366234688443104, 0.9375110953310847, 0.9382211965205042, 0.9391088230072785, 0.9389312977099237, 0.9393751109533108, 0.9390200603586011, 0.9395526362506658, 0.9393751109533108, 0.9409728386295048, 0.9399076868453755, 0.9421267530623114, 0.9415941771702467, 0.941239126575537, 0.941239126575537, 0.9427480916030534, 0.9413278892242144, 0.9420379904136339, 0.9431031421977631, 0.9432806674951181, 0.942659328954376, 0.942659328954376, 0.9431031421977631, 0.9423042783596662, 0.9441682939818924, 0.9432806674951181, 0.9432806674951181, 0.9418604651162791, 0.9439907686845376, 0.9439907686845376, 0.9436357180898278, 0.9446121072252796, 0.9446121072252796, 0.9445233445766021, 0.9438132433871826, 0.9446121072252796, 0.9454109710633766, 0.9443458192792473, 0.9458547843067637, 0.945499733712054, 0.9454109710633766, 0.9462098349014735, 0.945499733712054, 0.9459435469554411, 0.9460323096041185, 0.9457660216580863, 0.9466536481448606, 0.9464761228475058, 0.9465648854961832, 0.9463873601988283, 0.9470974613882478, 0.946919936090893, 0.945499733712054, 0.9470086987395704, 0.9466536481448606, 0.9474525119829575, 0.9477187999289899, 0.9473637493342801]
pos_dev_loss = [5091.813484553724, 4137.01253686251, 3689.477170485942, 3403.998225784806, 3210.7460914791623, 3060.975974822757, 2938.4697534657084, 2838.1272890002833, 2759.7365970159344, 2695.8446599066397, 2625.4102734431817, 2571.8545031162494, 2524.151691720798, 2493.387431870574, 2446.419346228078, 2402.7214218942463, 2365.752146972612, 2332.9018197482906, 2302.788246348632, 2269.8206814255345, 2243.765428676097, 2226.7345246415753, 2190.222103472334, 2164.7743008129255, 2169.9963956462857, 2125.975406064744, 2101.8390700101763, 2108.6721600800815, 2069.4122371772373, 2047.4137746377605, 2030.2733224395176, 2016.2643761664249, 1997.501385232471, 1986.6831390883556, 1976.447182973127, 1982.760600848443, 1941.2378157042244, 1929.1695167707355, 1919.9445141307751, 1909.1172952051975, 1899.507254019188, 1883.8044986582795, 1883.3727501317812, 1868.400715712623, 1856.262379875419, 1852.136649129316, 1846.073414917206, 1831.2357814963075, 1829.1762646242923, 1837.8617122136593, 1817.069175565305, 1800.7196462341624, 1813.3868609455067, 1790.9991845457994, 1788.8448377999634, 1781.0089124666922, 1768.7140906042844, 1763.4753082919917, 1758.2184054121062, 1752.1989765001144, 1746.6623988688827, 1746.3097051438795, 1734.3877564549807, 1730.075000390526, 1729.1523382919468, 1754.3297779890386, 1715.8782011746703, 1715.8692325939805, 1718.2523248982104, 1727.630246590706, 1700.2060423027006, 1696.4779204732677, 1702.1332309126778, 1685.7102744297295, 1681.9492569658057, 1675.7804216342981, 1693.7083847749407, 1668.6977594102639, 1665.2367292753574, 1670.5940970619256, 1657.190030982841, 1653.8817940872327, 1650.2453892192639, 1646.2048330719322, 1659.461737117778, 1641.211080451031, 1638.822761377738, 1636.8446743943464, 1628.7003779455881, 1627.8224232269513, 1636.3135534859798, 1627.8945464983817, 1616.2535474096787, 1614.9386265633734, 1661.4052228678252, 1607.0147550986771, 1611.6617283233716, 1602.2721888720039, 1600.9549727755782, 1597.793970235622]

epochs = np.arange(len(nonpos_dev_acc))
nonpos_test = plt.plot(np.arange(len(nonpos_new_test_loss)), nonpos_new_test_loss, label="nonpos Test Loss")
nonpos_dev = plt.plot(np.arange(len(nonpos_new_test_loss)), nonpos_dev_loss, label="nonpos Dev Loss")
pos_test = plt.plot(np.arange(len(pos_new_test_acc)), pos_new_test_loss, label="pos Test loss")
pos_dev = plt.plot(epochs, pos_dev_loss, label="pos dev loss")
plt.legend(["test_nonpos", "dev_nonpos", "test_pos", "dev_pos"])
plt.title("test and dev loss (with all words/pos)")
plt.savefig("test_dev_loss.png")
plt.legend()
plt.clf()

nonpos_test = plt.plot(np.arange(len(nonpos_new_test_acc)), nonpos_new_test_acc, label="nonpos Test Acc")
nonpos_dev = plt.plot(np.arange(len(nonpos_new_test_acc)), nonpos_dev_acc, label="nonpos Dev Acc")
pos_test = plt.plot(np.arange(len(pos_new_test_acc)), pos_new_test_acc, label="pos Test acc")
pos_dev = plt.plot(epochs, pos_dev_acc, label="pos dev acc")


plt.legend(["test_nonpos", "dev_nonpos", "test_pos", "dev_pos"])
plt.title("test and dev acc")
plt.savefig("test_dev_acc.png")
plt.clf()






