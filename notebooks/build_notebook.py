#!/usr/bin/env python3
"""Builds notebooks/adtc_qlora_benchmark.ipynb - the Colab-ready pre/post benchmark.
Run: python3 build_notebook.py
Output: adtc_qlora_benchmark.ipynb (self-contained, embeds eval + curated train data)
Also writes: data/reasoning_eval_questions.json, data/curated_train_pairs.json
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# ---------------------------------------------------------------------------
# 1. WAEC bilingual eval set (20 questions: 4 per language EN/YO/HA/SW/IG)
# ---------------------------------------------------------------------------
WAEC_EVAL = [
    # ---- English ----
    {"lang": "en", "subject": "math", "answer": "B",
     "question": "A trader buys a bag of rice for N24,000 and sells it for N30,000. What is the percentage profit?",
     "choices": {"A": "20%", "B": "25%", "C": "30%", "D": "15%"}},
    {"lang": "en", "subject": "math", "answer": "C",
     "question": "If 2x + 5 = 17, what is the value of x?",
     "choices": {"A": "4", "B": "5", "C": "6", "D": "7"}},
    {"lang": "en", "subject": "math", "answer": "B",
     "question": "The angles of a triangle are x, 2x and 3x. What is the value of x?",
     "choices": {"A": "20 degrees", "B": "30 degrees", "C": "40 degrees", "D": "60 degrees"}},
    {"lang": "en", "subject": "science", "answer": "B",
     "question": "Which of the following is NOT a vector quantity?",
     "choices": {"A": "Force", "B": "Mass", "C": "Velocity", "D": "Weight"}},
    # ---- Yoruba ----
    {"lang": "yo", "subject": "math", "answer": "B",
     "question": "Oníṣòwò kan ra àpò ìrẹsì kan ní N24,000, ó sì tà á ní N30,000. Kín ni ìpín ọgọ́rùn-ún èrè rẹ̀?",
     "choices": {"A": "20%", "B": "25%", "C": "30%", "D": "15%"}},
    {"lang": "yo", "subject": "math", "answer": "C",
     "question": "Bí 2x + 5 = 17, kín ni iye x?",
     "choices": {"A": "4", "B": "5", "C": "6", "D": "7"}},
    {"lang": "yo", "subject": "math", "answer": "B",
     "question": "Àwọn igun ìgbá mẹ́ta kan jẹ́ x, 2x àti 3x. Kín ni iye x?",
     "choices": {"A": "20 ìwọ̀n", "B": "30 ìwọ̀n", "C": "40 ìwọ̀n", "D": "60 ìwọ̀n"}},
    {"lang": "yo", "subject": "science", "answer": "B",
     "question": "Èwo nínú àwọn wọ̀nyí kì í ṣe òye afẹ̀sọ́nà (vector quantity)?",
     "choices": {"A": "Agbára", "B": "Ìwúwo", "C": "Ìyára", "D": "Àdánwó"}},
    # ---- Hausa ----
    {"lang": "ha", "subject": "math", "answer": "B",
     "question": "Wani ɗan kasuwa ya sayi buhun shinkafa akan N24,000 ya kuma sayar da shi akan N30,000. Menene kashi na riba?",
     "choices": {"A": "20%", "B": "25%", "C": "30%", "D": "15%"}},
    {"lang": "ha", "subject": "math", "answer": "C",
     "question": "Idan 2x + 5 = 17, menene darajar x?",
     "choices": {"A": "4", "B": "5", "C": "6", "D": "7"}},
    {"lang": "ha", "subject": "math", "answer": "B",
     "question": "Kusurwoyin alwatika sune x, 2x da 3x. Menene darajar x?",
     "choices": {"A": "20 digiri", "B": "30 digiri", "C": "40 digiri", "D": "60 digiri"}},
    {"lang": "ha", "subject": "science", "answer": "B",
     "question": "Wanne daga cikin waɗannan ba shi ne vector ba?",
     "choices": {"A": "Ƙarfi", "B": "Nauyi (mass)", "C": "Gudu", "D": "Nauyin da aka auna (weight)"}},
    # ---- Swahili ----
    {"lang": "sw", "subject": "math", "answer": "B",
     "question": "Mfanyabiashara alinunua gunia la mchele kwa N24,000 na kauza kwa N30,000. Ni asilimia ngapi ya faida?",
     "choices": {"A": "20%", "B": "25%", "C": "30%", "D": "15%"}},
    {"lang": "sw", "subject": "math", "answer": "C",
     "question": "Ikiwa 2x + 5 = 17, thamani ya x ni?",
     "choices": {"A": "4", "B": "5", "C": "6", "D": "7"}},
    {"lang": "sw", "subject": "math", "answer": "B",
     "question": "Pembe za pembetatu ni x, 2x na 3x. Thamani ya x ni?",
     "choices": {"A": "nyuzi 20", "B": "nyuzi 30", "C": "nyuzi 40", "D": "nyuzi 60"}},
    {"lang": "sw", "subject": "science", "answer": "B",
     "question": "Ni ipi kati ya hizi ambayo SI vector?",
     "choices": {"A": "Nguvu", "B": "Uzito (mass)", "C": "Kasi", "D": "Uzito (weight)"}},
    # ---- Igbo ----
    {"lang": "ig", "subject": "math", "answer": "B",
     "question": "Onye ahịa zụtara akpa osikapa na N24,000 wee ree ya na N30,000. Gịnị bụ pasentị uru ọ nwetara?",
     "choices": {"A": "20%", "B": "25%", "C": "30%", "D": "15%"}},
    {"lang": "ig", "subject": "math", "answer": "C",
     "question": "Ọ bụrụ na 2x + 5 = 17, gịnị bụ uru x?",
     "choices": {"A": "4", "B": "5", "C": "6", "D": "7"}},
    {"lang": "ig", "subject": "math", "answer": "B",
     "question": "Akụkụ nke triangle bụ x, 2x na 3x. Gịnị bụ uru x?",
     "choices": {"A": "digirii 20", "B": "digirii 30", "C": "digirii 40", "D": "digirii 60"}},
    {"lang": "ig", "subject": "science", "answer": "B",
     "question": "Olee nke n'ime ndị a na-abụghị vector quantity?",
     "choices": {"A": "Ike", "B": "Mass", "C": "Ọsọ", "D": "Ibu"}},
    {"lang": "en", "subject": "math", "answer": "B",
     "question": "A car travels at 60 km/h. How far will it go in 2.5 hours?",
     "choices": {"A": "120 km", "B": "150 km", "C": "180 km", "D": "200 km"}},
    {"lang": "yo", "subject": "math", "answer": "B",
     "question": "Ọkọ̀ ayọ́kẹ́lẹ́ kan ń rìn ní iyára 60 km/h. Báwo ni yóò ṣe jìna ní wákàtí 2.5?",
     "choices": {"A": "120 km", "B": "150 km", "C": "180 km", "D": "200 km"}},
    {"lang": "ha", "subject": "math", "answer": "B",
     "question": "Wata mota tana tafiya da gudu 60 km/h. Nawa za ta tafi cikin awanni 2.5?",
     "choices": {"A": "120 km", "B": "150 km", "C": "180 km", "D": "200 km"}},
    {"lang": "sw", "subject": "math", "answer": "B",
     "question": "Gari inatembea kwa kasi 60 km/h. Itafikia mbali gani kwa saa 2.5?",
     "choices": {"A": "120 km", "B": "150 km", "C": "180 km", "D": "200 km"}},
    {"lang": "ig", "subject": "math", "answer": "B",
     "question": "Otu ụgbọ ala na-eme 60 km/h. Kedu anya ọ ga-eme n'ime awa 2.5?",
     "choices": {"A": "120 km", "B": "150 km", "C": "180 km", "D": "200 km"}}

]

# ---------------------------------------------------------------------------
# 2. Curated bilingual training pairs (step-by-step, EN + YO + IG)
# ---------------------------------------------------------------------------
CURATED_TRAIN = [
  {
    "lang": "en",
    "instruction": "A solar panel yields 2 kWh on a sunny day and 0.5 kWh on a cloudy day. If the next three days are sunny, cloudy, sunny, estimate the total energy output and show your working step by step.",
    "response": "Step 1: Find the profit.\n$\\text{Profit} = \\text{selling price} - \\text{cost price} = \\text{N}30{,}000 - \\text{N}24{,}000 = \\text{N}6{,}000$.\nStep 2: Find the percentage profit.\n$\\text{Percentage profit} = \\frac{\\text{profit}}{\\text{cost price}} \\times 100 = \\frac{6{,}000}{24{,}000} \\times 100$.\nStep 3: Simplify: $\\frac{6{,}000}{24{,}000} = 0.25$. Multiply by 100: $0.25 \\times 100 = 25$.\nAnswer: The percentage profit is $25\%$."
  },
  {
    "lang": "yo",
    "instruction": "Pánẹ́lì òòrùn kan ń pèsè 2 kWh ní ọjọ́ oòrùn, ó sì ń pèsè 0.5 kWh ní ọjọ́ ìkùukùu. Bí ọjọ́ mẹ́ta tó máa bọ̀ bá jẹ́ oòrùn, ìkùukùu, oòrùn, ṣe ìṣirò àpapọ̀ agbára tó máa jáde, kí o sì ṣàlàyé ìgbésẹ̀ kọ̀ọ̀kan.",
    "response": "Ìgbésẹ̀ 1: Rí èrè.\n$\\text{Èrè} = \\text{iye tí a tà á} - \\text{iye tí a ra á} = \\text{N}30{,}000 - \\text{N}24{,}000 = \\text{N}6{,}000$.\nÌgbésẹ̀ 2: Rí ìpín ọgọ́rùn-ún èrè.\n$\\text{Ìpín ọgọ́rùn-ún èrè} = \\frac{\\text{èrè}}{\\text{iye tí a ra á}} \\times 100 = \\frac{6{,}000}{24{,}000} \\times 100$.\nÌgbésẹ̀ 3: Rọrùn rẹ́: $\\frac{6{,}000}{24{,}000} = 0.25$. Fi 100 sọ̀: $0.25 \\times 100 = 25$.\nÌdáhùn: Ìpín ọgọ́rùn-ún èrè jẹ́ $25\%$."
  },
  {
    "lang": "en",
    "instruction": "If 2x + 5 = 17, what is the value of x? Show your working.",
    "response": "Step 1: Subtract 5 from both sides: $2x + 5 - 5 = 17 - 5$, so $2x = 12$.\nStep 2: Divide both sides by 2: $\\frac{2x}{2} = \\frac{12}{2}$, so $x = 6$.\nAnswer: $x = 6$."
  },
  {
    "lang": "yo",
    "instruction": "Bí 2x + 5 = 17, kín ni iye x? Ṣàlàyé ìgbésẹ̀ rẹ.",
    "response": "Ìgbésẹ̀ 1: Yọ 5 kúrò ní ẹ̀gbẹ́ méjèèjì: $2x + 5 - 5 = 17 - 5$, nítorí náà $2x = 12$.\nÌgbésẹ̀ 2: Pín 2 ní ẹ̀gbẹ́ méjèèjì: $\\frac{2x}{2} = \\frac{12}{2}$, nítorí náà $x = 6$.\nÌdáhùn: $x = 6$."
  },
  {
    "lang": "en",
    "instruction": "The angles of a triangle are x, 2x and 3x. Find the value of x.",
    "response": "Step 1: The sum of angles in a triangle is $180^\\circ$.\nStep 2: So $x + 2x + 3x = 180$, which means $6x = 180$.\nStep 3: Divide both sides by 6: $x = \\frac{180}{6} = 30$.\nAnswer: $x = 30^\\circ$."
  },
  {
    "lang": "yo",
    "instruction": "Àwọn igun ìgbá mẹ́ta kan jẹ́ x, 2x àti 3x. Rí iye x.",
    "response": "Ìgbésẹ̀ 1: Àpapọ̀ igun ìgbá mẹ́ta jẹ́ $180^\\circ$.\nÌgbésẹ̀ 2: Nítorí náà $x + 2x + 3x = 180$, èyí túmọ̀ sí $6x = 180$.\nÌgbésẹ̀ 3: Pín 6 ní ẹ̀gbẹ́ méjèèjì: $x = \\frac{180}{6} = 30$.\nÌdáhùn: $x = 30^\\circ$."
  },
  {
    "lang": "en",
    "instruction": "A student scored 45 out of 60 in a test. What percentage did she score?",
    "response": "Step 1: Percentage $= \\frac{\\text{score}}{\\text{total}} \\times 100 = \\frac{45}{60} \\times 100$.\nStep 2: Simplify: $\\frac{45}{60} = 0.75$. Multiply by 100: $0.75 \\times 100 = 75$.\nAnswer: She scored $75\%$."
  },
  {
    "lang": "yo",
    "instruction": "Ọmọkọ́lẹ̀bíì kan gba 45 nínú 60 nínú ìdánwò kan. Ìpín ọgọ́rùn-ún mélòó ni ó gbà?",
    "response": "Ìgbésẹ̀ 1: Ìpín ọgọ́rùn-ún $= \\frac{\\text{àmì tí ó gbà}}{\\text{àpapọ̀}} \\times 100 = \\frac{45}{60} \\times 100$.\nÌgbésẹ̀ 2: Rọrùn rẹ́: $\\frac{45}{60} = 0.75$. Fi 100 sọ̀: $0.75 \\times 100 = 75$.\nÌdáhùn: Ó gba $75\%$."
  },
  {
    "lang": "en",
    "instruction": "What is 15% of 200?",
    "response": "Step 1: $15\%$ means 15 out of 100, which is $\\frac{15}{100} = 0.15$.\nStep 2: Multiply: $0.15 \\times 200 = 30$.\nAnswer: $15\%$ of 200 is $30$."
  },
  {
    "lang": "yo",
    "instruction": "Kín ni 15% nínú 200?",
    "response": "Ìgbésẹ̀ 1: $15\%$ túmọ̀ sí 15 nínú 100, èyí tí ó jẹ́ $\\frac{15}{100} = 0.15$.\nÌgbésẹ̀ 2: Ìsọdipúpọ̀: $0.15 \\times 200 = 30$.\nÌdáhùn: $15\%$ nínú 200 jẹ́ $30$."
  },
  {
    "lang": "en",
    "instruction": "A shop gives a 10% discount on a radio priced at N8,000. How much does the buyer pay?",
    "response": "Step 1: Find the discount.\n$\\text{Discount} = 10\% \\text{ of } \\text{N}8{,}000 = 0.10 \\times 8{,}000 = \\text{N}800$.\nStep 2: Subtract from the price: $\\text{N}8{,}000 - \\text{N}800 = \\text{N}7{,}200$.\nAnswer: The buyer pays $\\text{N}7{,}200$."
  },
  {
    "lang": "yo",
    "instruction": "Ilé ìtajà kan fún ni 10% ìdínkù lórí rédíò kan tí iye rẹ̀ jẹ́ N8,000. Eléyìí mélòó ni olùrà á san?",
    "response": "Ìgbésẹ̀ 1: Rí ìdínkù.\n$\\text{Ìdínkù} = 10\% \\text{ nínú } \\text{N}8{,}000 = 0.10 \\times 8{,}000 = \\text{N}800$.\nÌgbésẹ̀ 2: Yọ ọ́ kúrò nínú iye: $\\text{N}8{,}000 - \\text{N}800 = \\text{N}7{,}200$.\nÌdáhùn: Olùrà á san $\\text{N}7{,}200$."
  },
  {
    "lang": "ig",
    "instruction": "Onye ahịa zụtara akpa osikapa na N24,000 wee ree ya na N30,000. Gbakọọ pasentị uru ọ nwetara, gosi usoro gị nke ọ bụla.",
    "response": "Nzọụkwụ 1: Chọta uru.\n$\\text{Uru} = \\text{ọnụahịa e rere} - \\text{ọnụahịa azụrụ} = \\text{N}30{,}000 - \\text{N}24{,}000 = \\text{N}6{,}000$.\nNzọụkwụ 2: Chọta pasentị uru.\n$\\text{Pasentị uru} = \\frac{\\text{uru}}{\\text{ọnụahịa azụrụ}} \\times 100 = \\frac{6{,}000}{24{,}000} \\times 100$.\nNzọụkwụ 3: Mee ka ọ dị mfe: $\\frac{6{,}000}{24{,}000} = 0.25$. Mụbaa na 100: $0.25 \\times 100 = 25$.\nAzịza: Pasentị uru bụ $25\%$."
  },
  {
    "lang": "ig",
    "instruction": "Ọ bụrụ na 2x + 5 = 17, gịnị bụ uru x? Gosi ọrụ gị.",
    "response": "Nzọụkwụ 1: Wepụ 5 n'akụkụ abụọ: $2x + 5 - 5 = 17 - 5$, yabụ $2x = 12$.\nNzọụkwụ 2: Kewaa akụkụ abụọ site na 2: $\\frac{2x}{2} = \\frac{12}{2}$, yabụ $x = 6$.\nAzịza: $x = 6$."
  },
  {
    "lang": "ha",
    "instruction": "Wani ɗan kasuwa ya sayi buhun shinkafa kan N24,000 ya kuma sayar da shi kan N30,000. Lissafa kashi na riba a cikin dari, ka nuna yadda aka same shi mataki-mataki.",
    "response": "Mataki 1: Nemo riba.\n$\\text{Riba} = \\text{farashin sayarwa} - \\text{farashin siye} = \\text{N}30{,}000 - \\text{N}24{,}000 = \\text{N}6{,}000$.\nMataki 2: Nemo kashi na riba.\n$\\text{Kashi na riba} = \\frac{\\text{riba}}{\\text{farashin siye}} \\times 100 = \\frac{6{,}000}{24{,}000} \\times 100$.\nMataki 3: Sauƙaƙe: $\\frac{6{,}000}{24{,}000} = 0.25$. Ninka da 100: $0.25 \\times 100 = 25$.\nAmsa: Kashi na riba shine $25\%$."
  },
  {
    "lang": "ha",
    "instruction": "Idan 2x + 5 = 17, menene darajar x? Nuna yadda aka same shi.",
    "response": "Mataki 1: Cire 5 daga bangarori biyu: $2x + 5 - 5 = 17 - 5$, don haka $2x = 12$.\nMataki 2: Raba bangarori biyu da 2: $\\frac{2x}{2} = \\frac{12}{2}$, don haka $x = 6$.\nAmsa: $x = 6$."
  },
  {
    "lang": "sw",
    "instruction": "Mfanyabiashara alinunua gunia la mchele kwa N24,000 na kauza kwa N30,000. Tuma asilimia ya faida na kuonyesha hatua kwa hatua.",
    "response": "Hatua 1: Pata faida.\n$\\text{Faida} = \\text{bei ya mauzo} - \\text{bei ya ununuzi} = \\text{N}30{,}000 - \\text{N}24{,}000 = \\text{N}6{,}000$.\nHatua 2: Pata asilimia ya faida.\n$\\text{Asilimia ya faida} = \\frac{\\text{faida}}{\\text{bei ya ununuzi}} \\times 100 = \\frac{6{,}000}{24{,}000} \\times 100$.\nHatua 3: Rahisisha: $\\frac{6{,}000}{24{,}000} = 0.25$. Zidisha kwa 100: $0.25 \\times 100 = 25$.\nJibu: Asilimia ya faida ni $25\%$."
  },
  {
    "lang": "sw",
    "instruction": "Ikiwa 2x + 5 = 17, thamani ya x ni? Onyesha jinsi ulivyopata.",
    "response": "Hatua 1: Ondoa 5 pande zote: $2x + 5 - 5 = 17 - 5$, kwa hiyo $2x = 12$.\nHatua 2: Gawa pande zote kwa 2: $\\frac{2x}{2} = \\frac{12}{2}$, kwa hiyo $x = 6$.\nJibu: $x = 6$."
  },
  {
    "lang": "en",
    "instruction": "A bag contains 3 red marbles and 5 blue marbles. What fraction of the marbles are red? Show your working.",
    "response": "Step 1: Find the total number of marbles.\n$$\\text{Total} = 3 + 5 = 8$$\nStep 2: The red marbles are 3 out of 8.\n$$\\text{Fraction red} = \\frac{3}{8}$$\nAnswer: $\\frac{3}{8}$ of the marbles are red."
  },
  {
    "lang": "yo",
    "instruction": "Àpò kan ní àwọn òkúta 3 pupa àti 5 búlúù. Ìpín mélòó nínú àwọn òkúta ni ó pupa? Ṣàlàyé ìgbésẹ̀ rẹ.",
    "response": "Ìgbésẹ̀ 1: Rí àpapọ̀ àwọn òkúta.\n$$\\text{Àpapọ̀} = 3 + 5 = 8$$\nÌgbésẹ̀ 2: Àwọn òkúta pupa jẹ́ 3 nínú 8.\n$$\\text{Ìpín pupa} = \\frac{3}{8}$$\nÌdáhùn: $\\frac{3}{8}$ nínú àwọn òkúta ni ó pupa."
  },
  {
    "lang": "ig",
    "instruction": "Akpa nwere okwute uhie 3 na nke anunu 5. Gosi puru ole n'ime okwute bu uhie? Gosi usoro gi.",
    "response": "Nzọụkwụ 1: Chọta ọnụọgụgụ okwute niile.\n$$\\text{Àpapọ̀} = 3 + 5 = 8$$\nNzọụkwụ 2: Okwute uhie bụ 3 n'ime 8.\n$$\\text{Ìpín uhie} = \\frac{3}{8}$$\nAzịza: $\\frac{3}{8}$ n'ime okwute bụ uhie."
  },
  {
    "lang": "ha",
    "instruction": "Wani jaka yana dauke da duwatsu masu ja 3 da kuma shuɗi 5. Wace kashi na duwatsun ya kasance ja? Nuna yadda aka same shi.",
    "response": "Mataki 1: Nemo jimlar adadin duwatsu.\n$$\\text{Jimla} = 3 + 5 = 8$$\nMataki 2: Duwatsun ja su ne 3 cikin 8.\n$$\\text{Kashi ja} = \\frac{3}{8}$$\nAmsa: $\\frac{3}{8}$ na duwatsun ya kasance ja."
  },
  {
    "lang": "sw",
    "instruction": "Mfuko ana mawe 3 mekundu na 5 ya bluu. Sehemu gani ya mawe ni mekundu? Onyesha jinsi ulivyopata.",
    "response": "Hatua 1: Pata jumla ya mawe.\n$$\\text{Jumla} = 3 + 5 = 8$$\nHatua 2: Mawe mekundu ni 3 kati ya 8.\n$$\\text{Sehemu mekundu} = \\frac{3}{8}$$\nJibu: $\\frac{3}{8}$ ya mawe ni mekundu."
  },
  {
    "lang": "en",
    "instruction": "A bus travels 60 km in 1.5 hours. What is its average speed in km/h, and how long would it take to cover 150 km at that speed? Show your working.",
    "response": "Step 1: Find the average speed.\nSpeed = distance / time = 60 / 1.5 = 40.\nStep 2: The average speed is 40 km/h.\nStep 3: Find the time for 150 km.\nTime = distance / speed = 150 / 40 = 3.75.\nAnswer: Average speed is 40 km/h, and it takes 3.75 hours (3 hours 45 minutes) to cover 150 km."
  },
  {
    "lang": "yo",
    "instruction": "Ọkọ̀ akérò kan ń rìn 60 km ní 1.5 wákàtí. Kí ni iyára àpapọ̀ rẹ̀ ní km/h, tó sì máa gba àkókò tó pé tó rìn 150 km ní iyára yẹn? Ṣàlàyé ìgbésẹ̀ rẹ̀.",
    "response": "Ìgbésẹ̀ 1: Rí iyára àpapọ̀.\nIyára = bí ṣe jìna / àkókò = 60 / 1.5 = 40.\nÌgbésẹ̀ 2: Iyára àpapọ̀ jẹ́ 40 km/h.\nÌgbésẹ̀ 3: Rí àkókò fún 150 km.\nÀkókò = bí ṣe jìna / iyára = 150 / 40 = 3.75.\nÌdáhùn: Iyára àpapọ̀ jẹ́ 40 km/h, ó sì máa gba wákàtí 3.75 (wákàtí 3 àti ìṣẹ́jú 45) láti rìn 150 km."
  },
  {
    "lang": "ha",
    "instruction": "Wata mota tana tafiya 60 km a cikin awanni 1.5. Menene matsakaitan gudu a km/h, kuma tsawon lokacin da za ta ɗauka don tafiyar 150 km a wannan gudu? Nuna yadda aka same shi.",
    "response": "Mataki 1: Nemo matsakaitan gudu.\nGudu = nisa / lokaci = 60 / 1.5 = 40.\nMataki 2: Matsakaitan gudu shine 40 km/h.\nMataki 3: Nemo lokacin 150 km.\nLokaci = nisa / gudu = 150 / 40 = 3.75.\nAmsa: Matsakaitan gudu shine 40 km/h, kuma yana ɗaukar awanni 3.75 (awanni 3 da mintoci 45) don tafiyar 150 km."
  },
  {
    "lang": "sw",
    "instruction": "Bas inasafiri 60 km kwa saa 1.5. Ni kasi ya wastani yake kwa km/h, na muda gani itachukua kufikia 150 km kwa kasi hiyo? Onyesha jinsi ulivyopata.",
    "response": "Hatua 1: Pata kasi ya wastani.\nKasi = mbali / muda = 60 / 1.5 = 40.\nHatua 2: Kasi ya wastani ni 40 km/h.\nHatua 3: Pata muda wa 150 km.\nMuda = mbali / kasi = 150 / 40 = 3.75.\nJibu: Kasi ya wastani ni 40 km/h, na inachukua saa 3.75 (saa 3 na dakika 45) kufikia 150 km."
  },
  {
    "lang": "ig",
    "instruction": "Otu ụgbọ ala na-eme 60 km n'ime awa 1.5. Gịnị bụ oke ọsọ ọ na-eme n'otu elekere (km/h), ma oge ole ọ ga-ewe ya iki gafere 150 km n'ọsọ ahụ? Gosi usoro gị.",
    "response": "Nzọụkwụ 1: Chọta oke ọsọ.\nỌsọ = ewere / oge = 60 / 1.5 = 40.\nNzọụkwụ 2: Oke ọsọ bụ 40 km/h.\nNzọụkwụ 3: Chọta oge maka 150 km.\nOge = ewere / ọsọ = 150 / 40 = 3.75.\nAzịza: Oke ọsọ bụ 40 km/h, ọ ga-ewere awa 3.75 (awa 3 na nkeji 45) iki gafere 150 km."
  },
  {
    "lang": "en",
    "instruction": "A town has 2,000 people and grows by 10% each year. Estimate the population after 2 years. Show your working.",
    "response": "Step 1: The growth factor per year is 1 + 0.10 = 1.10.\nStep 2: After 2 years, population = 2000 x (1.10)^2.\nStep 3: (1.10)^2 = 1.21, so 2000 x 1.21 = 2420.\nAnswer: The population after 2 years is about 2,420."
  },
  {
    "lang": "yo",
    "instruction": "Ìlú kan ní ènìyàn 2,000, ó sì ń pọ̀ sí 10% lọ́dún. Ṣe ìṣirò iye ènìyàn tó máa jẹ́ lẹ́yìn ọdún méjì. Ṣàlàyé ìgbésẹ̀ rẹ̀.",
    "response": "Ìgbésẹ̀ 1: Ìpín ìdàgbàsókè lọ́dún jẹ́ 1 + 0.10 = 1.10.\nÌgbésẹ̀ 2: Lẹ́yìn ọdún méjì, iye ènìyàn = 2000 x (1.10)^2.\nÌgbésẹ̀ 3: (1.10)^2 = 1.21, nítorí náà 2000 x 1.21 = 2420.\nÌdáhùn: Iye ènìyàn lẹ́yìn ọdún méjì jẹ́ nǹkan bí 2,420."
  },
  {
    "lang": "ha",
    "instruction": "Wani gari yana da mutane 2,000, kuma yana karuwa da 10% a kowace shekara. Kimanta yawan jama'a bayan shekaru 2. Nuna yadda aka same shi.",
    "response": "Mataki 1: Mahimmin karuwa a shekara shine 1 + 0.10 = 1.10.\nMataki 2: Bayan shekaru 2, yawan jama'a = 2000 x (1.10)^2.\nMataki 3: (1.10)^2 = 1.21, don haka 2000 x 1.21 = 2420.\nAmsa: Yawan jama'a bayan shekaru 2 ya kai kimanin 2,420."
  },
  {
    "lang": "sw",
    "instruction": "Mji una watu 2,000, na ukua kwa 10% kila mwaka. Kadilia idadi ya watu baada ya miaka 2. Onyesha jinsi ulivyopata.",
    "response": "Hatua 1: Sababu ya ukuaji kwa mwaka ni 1 + 0.10 = 1.10.\nHatua 2: Baada ya miaka 2, idadi = 2000 x (1.10)^2.\nHatua 3: (1.10)^2 = 1.21, kwa hiyo 2000 x 1.21 = 2420.\nJibu: Idadi ya watu baada ya miaka 2 ni takriban 2,420."
  }

]

# ---------------------------------------------------------------------------
# Helper: build a notebook code cell
# ---------------------------------------------------------------------------
def code(src, exec_count=None):
    return {"cell_type": "code", "execution_count": exec_count, "id": "c" + str(abs(hash(src)) % 10**8),
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "id": "m" + str(abs(hash(src)) % 10**8),
            "metadata": {}, "source": src.splitlines(keepends=True)}

# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------
C_INSTALL = r'''
# Cell 1: install dependencies (run once)
# tiny-aya-global needs a recent transformers (cohere2 arch support).
!pip install -q -U "transformers>=4.46" "peft>=0.14" "trl>=0.13,<0.16" bitsandbytes accelerate datasets matplotlib seaborn

# Check GPU (free Colab gives T4, ~15 GB VRAM - plenty for a 3.3B 4-bit model)
!nvidia-smi
'''

C_LOGIN = r'''
# Cell 2: Hugging Face login (REQUIRED once)
#
# The base model CohereLabs/tiny-aya-global is access-restricted.
# 1. Open https://huggingface.co/CohereLabs/tiny-aya-global in your browser
# 2. Click "Agree and access repository" (one click, instant)
# 3. The token is read from the HF_TOKEN environment variable (set it in Colab
#    before running this cell). Create one: https://huggingface.co/settings/tokens
#    (READ scope for the base model; WRITE scope if you also run the export Cell 15).
from huggingface_hub import login, notebook_login
_TOKEN = "__HF_TOKEN__"
try:
    if _TOKEN.startswith("hf_"):
        login(token=_TOKEN, add_to_git_credential=False)
        print("Logged in with embedded token.")
    else:
        notebook_login()
except Exception as e:
    print("Auto-login failed:", e)
    notebook_login()
'''

C_CONFIG = r'''
# Cell 3: configuration
import json, os, re, random, gc
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "CohereLabs/tiny-aya-global"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# arc_easy eval size (the ADTC profiler uses 50 by default; real audit uses more)
ARC_N = 50
ARC_SHOT = 5
MAX_NEW = 128

OUT = "/content/adtc_results"
os.makedirs(OUT, exist_ok=True)
print("config ok")
'''

C_LOAD = r'''
# Cell 4: load base model in 4-bit (QLoRA-ready)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.float16)
except Exception as e:
    print("First load failed (", type(e).__name__, "), retrying with trust_remote_code=True")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.float16,
                                                 trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("loaded:", MODEL_ID)
print("params trainable-check: model is 4-bit base, ready for LoRA")
'''

C_EVALHELP = r'''
# Cell 5: evaluation helpers (arc_easy + WAEC bilingual set)
# arc_easy is scored exactly like lm-eval does for the ADTC profiler:
#   5-shot, loglikelihood over answer letters A-D, argmax wins.
# We also report a free-generation variant (greedy decode + letter extraction).

def build_arc_prompt(q, choices, shots):
    lines = ["The following are multiple choice questions (with answers) about science."]
    for s in shots:
        sq, sch, sans = s
        lines.append("")
        lines.append(f"Question: {sq}")
        lines.append("Choices:")
        for L in "ABCD":
            lines.append(f"{L}. {sch[L]}")
        lines.append(f"Answer: {sans}")
    lines.append("")
    lines.append(f"Question: {q}")
    lines.append("Choices:")
    for L in "ABCD":
        lines.append(f"{L}. {choices[L]}")
    lines.append("Answer:")
    return "\n".join(lines)

def letters_loglik(model, tok, prompt):
    """Score each letter A-D appended after the prompt; return argmax letter + probs."""
    base_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    probs = {}
    with torch.no_grad():
        for L in "ABCD":
            ids = torch.cat([base_ids, tok(" " + L, return_tensors="pt").input_ids.to(model.device)], dim=1)
            out = model(ids).logits[0, -2:-1, :]
            logp = torch.log_softmax(out, dim=-1)[0, ids[0, -1]].item()
            probs[L] = logp
    return max(probs, key=probs.get), probs

def generate_answer(model, tok, prompt, max_new=MAX_NEW):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def extract_letter(text):
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None

def eval_arc(model, tok, train_rows, test_rows, n=ARC_N, shots=ARC_SHOT):
    test_rows = test_rows[:n]
    ll_correct = 0; gen_correct = 0
    for i, row in enumerate(test_rows):
        prompt = build_arc_prompt(row["question"], row["choices"], train_rows[:shots])
        letter_ll, _ = letters_loglik(model, tok, prompt)
        if letter_ll == row["answer"]: ll_correct += 1
        gen = generate_answer(model, tok, prompt)
        letter_gen = extract_letter(gen)
        if letter_gen == row["answer"]: gen_correct += 1
        if (i + 1) % 10 == 0:
            print(f"  arc_easy {i+1}/{n}  ll_acc={ll_correct/(i+1):.3f}  gen_acc={gen_correct/(i+1):.3f}")
    return {"arc_easy_ll_acc": ll_correct / n, "arc_easy_gen_acc": gen_correct / n, "n": n}

def chat_prompt(question, choices=None):
    if choices:
        q = question + "\n\nChoices:\n" + "\n".join(f"{L}. {choices[L]}" for L in "ABCD")
    else:
        q = question
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)

def eval_waec(model, tok, items):
    results = []
    for it in items:
        prompt = chat_prompt(it["question"], it["choices"])
        gen = generate_answer(model, tok, prompt)
        letter = extract_letter(gen)
        correct = letter == it["answer"]
        results.append({**it, "generated": gen, "letter": letter, "correct": correct})
    return results

def waec_summary(results):
    langs = {}
    for r in results:
        langs.setdefault(r["lang"], {"n": 0, "c": 0})
        langs[r["lang"]]["n"] += 1
        langs[r["lang"]]["c"] += 1 if r["correct"] else 0
    return {k: round(v["c"] / v["n"], 3) for k, v in langs.items()}, \
           round(sum(1 for r in results if r["correct"]) / len(results), 3)
'''

C_ARCDATA = r'''
# Cell 6: load arc_easy (same source lm-eval uses: allenai/ai2_arc)
arc = load_dataset("allenai/ai2_arc", "ARC-Easy")
def fmt(row):
    q = row["question"]
    ch = {L: row["choices"]["text"][i] for i, L in enumerate("ABCD")}
    return {"question": q, "choices": ch, "answer": row["answerKey"]}
train_rows = [fmt(r) for r in arc["train"]][:ARC_SHOT]
random.shuffle(arc["train"])
test_rows = [fmt(r) for r in arc["test"]]
random.seed(SEED); random.shuffle(test_rows)
print(f"arc_easy loaded: {len(test_rows)} test rows, {len(train_rows)} shots")
'''

C_BASE_EVAL = r'''
# Cell 7: BASELINE (base model, before fine-tuning)
print("== arc_easy (base) ==")
base_arc = eval_arc(model, tok, train_rows, test_rows, n=ARC_N)
print("baseline arc_easy:", base_arc)
with open(f"{OUT}/base_arc.json", "w") as f: json.dump(base_arc, f, indent=2)
'''

C_BASE_WAEC = r'''
# Cell 8: BASELINE on the bilingual reasoning set (EN/YO/HA/SW/IG)
WAEC_EVAL = json.loads("__WAEC_EVAL__")
print("== WAEC bilingual set (base) ==")
base_waec = eval_waec(model, tok, WAEC_EVAL)
by_lang, overall = waec_summary(base_waec)
print("by language:", by_lang, "| overall:", overall)
with open(f"{OUT}/base_waec.json", "w") as f:
    json.dump({"by_lang": by_lang, "overall": overall, "items": base_waec}, f, indent=2, ensure_ascii=False)
'''

C_BASE_DEMO = r'''
# Cell 9: BASELINE judge demo - the 2 exact test prompts from metadata.json
DEMO_PROMPTS = [
    ("tp_001 (EN)", "A trader buys a bag of rice for N24,000 and sells it for N30,000. Calculate the percentage profit and show your working step by step."),
    ("tp_002 (YO)", "Oníṣòwò kan ra àpò ìrẹsì kan ní N24,000, ó sì tà á ní N30,000. Ṣe ìṣirò ìpín ọgọ́rùn-ún èrè rẹ̀, kí o sì ṣàlàyé ìgbésẹ̀ kọ̀ọ̀kan."),
]
print("== judge demo (base) ==")
base_demo = {}
for pid, p in DEMO_PROMPTS:
    out = generate_answer(model, tok, chat_prompt(p), max_new=256)
    base_demo[pid] = out
    print(f"\n--- {pid} ---\n{out}")
with open(f"{OUT}/base_demo.json", "w") as f:
    json.dump(base_demo, f, indent=2, ensure_ascii=False)
'''

C_TRAIN_HEADER = r'''
# =====================================================================
#  FINE-TUNING (QLoRA) - the math and scientific-reasoning experiment
#  Training sources (all verified on Hugging Face):
#   - masakhane/afrimgsm: GSM8k math word problems in 16+ African
#     languages (yor, hau, swa, ibo, eng used here) - core math corpus
#   - masakhane/afrimmlu: MMLU translated into African languages + English
#     (yor/hau/swa/ibo/eng) - broad math + science reasoning, multilingual
#   - cais/mmlu (STEM subjects): deep English math + science coverage
#   - openai/gsm8k: extra English math word problems
#   - worldboss/waec-integrated-science-2007: real WAEC past questions
#   - honourjesus/nllb-hausa-waec-translations: WAEC content in Hausa
#   - curated bilingual step-by-step pairs (embedded in this notebook)
#  Each source is optional: if one fails to download, training continues
#  with the rest (set TRAIN_CAPS below to control runtime on T4).
# =====================================================================
'''

C_TRAINDATA = r'''
# Cell 10: build the training dataset
TRAIN_CAPS = {"afrimgsm_yor": 300, "afrimgsm_hau": 300, "afrimgsm_swa": 300,
              "afrimgsm_ibo": 300, "afrimgsm_eng": 150, "waec2007": 29, "hausa_waec": 300,
              "afrimmlu_eng": 150, "afrimmlu_yor": 150, "afrimmlu_hau": 150,
              "afrimmlu_swa": 150, "afrimmlu_ibo": 150, "mmlu_stem": 1500, "gsm8k_eng": 400}
CURATED_TRAIN = json.loads("__CURATED_TRAIN__")

def to_msgs(instruction, response):
    return [{"role": "user", "content": instruction},
            {"role": "assistant", "content": response}]

all_pairs = []
for lang, ins, resp in [(p["lang"], p["instruction"], p["response"]) for p in CURATED_TRAIN]:
    all_pairs.append(to_msgs(ins, resp))
print(f"curated pairs: {len(all_pairs)}")

# afrimgsm (per-language configs: yor, hau, swa, ibo, eng)
for cfg, cap in [("yor", TRAIN_CAPS["afrimgsm_yor"]), ("hau", TRAIN_CAPS["afrimgsm_hau"]),
                 ("swa", TRAIN_CAPS["afrimgsm_swa"]), ("ibo", TRAIN_CAPS["afrimgsm_ibo"]),
                 ("eng", TRAIN_CAPS["afrimgsm_eng"])]:
    try:
        ds = load_dataset("masakhane/afrimgsm", cfg, split="train")
        n = 0
        for row in ds:
            if n >= cap: break
            all_pairs.append(to_msgs("Solve this mathematics problem step by step.\n\n" + row["question"],
                                     row["answer"]))
            n += 1
        print(f"afrimgsm [{cfg}]: {n}")
    except Exception as e:
        print(f"afrimgsm [{cfg}] SKIPPED: {type(e).__name__}: {str(e)[:120]}")

# WAEC integrated science 2007 (real past questions)
try:
    ds = load_dataset("worldboss/waec-integrated-science-2007", split="train")
    n = 0
    for row in ds:
        if n >= TRAIN_CAPS["waec2007"]: break
        q = row.get("Question") or row.get("question")
        a = row.get("Answer") or row.get("answer")
        if q and a:
            all_pairs.append(to_msgs("Answer this WAEC science question.\n\n" + str(q), str(a)))
            n += 1
    print(f"waec-integrated-science-2007: {n}")
except Exception as e:
    print(f"waec2007 SKIPPED: {type(e).__name__}: {str(e)[:120]}")

# Hausa WAEC translations
try:
    ds = load_dataset("honourjesus/nllb-hausa-waec-translations", split="train")
    cols = ds.column_names
    qcol = next((c for c in cols if "question" in c.lower() or "text" in c.lower() or "en" in c.lower()), cols[0] if cols else None)
    acol = next((c for c in cols if "answer" in c.lower() or "ha" in c.lower() or "trans" in c.lower()), None)
    n = 0
    for row in ds:
        if n >= TRAIN_CAPS["hausa_waec"]: break
        q, a = row.get(qcol), (row.get(acol) if acol else None)
        if q and a:
            all_pairs.append(to_msgs("Amsa wannan tambayar jarrabawa (WAEC).\n\n" + str(q), str(a)))
            n += 1
    print(f"hausa-waec: {n}")
except Exception as e:
    print(f"hausa_waec SKIPPED: {type(e).__name__}: {str(e)[:120]}")

# ---- BROADER MATH + SCIENCE CORPORA (added for generalization) ----
# AfriMMLU: MMLU translated into African languages + English. Covers mathematics,
# physics, chemistry, biology and more across yor/hau/swa/ibo/eng.
AFRIMMLU_LANGS = {"eng": "eng_Latn", "yor": "yor_Latn", "hau": "hau_Latn",
                  "swa": "swa_Latn", "ibo": "ibo_Latn"}
for lang, cfg in AFRIMMLU_LANGS.items():
    try:
        ds = load_dataset("masakhane/afrimmlu", cfg, split="train")
        cap = TRAIN_CAPS.get(f"afrimmlu_{lang}", 150)
        n = 0
        for row in ds:
            if n >= cap: break
            q = row.get("question") or row.get("Question")
            ans = row.get("answer") or row.get("Answer")
            choices = row.get("choices") or row.get("Choices")
            if q is None or ans is None: continue
            if choices:
                try:
                    cidx = "ABCD".index(ans) if ans in "ABCD" else int(ans)
                    opt = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices))
                    resp = f"The correct option is {'ABCD'[cidx]}.\n\n{choices[cidx]}"
                    all_pairs.append(to_msgs(f"Answer this question.\n\n{q}\n\n{opt}", resp))
                    n += 1; continue
                except Exception:
                    pass
            all_pairs.append(to_msgs(f"Answer this question.\n\n{q}", str(ans))); n += 1
        print(f"afrimmlu [{lang}]: {n}")
    except Exception as e:
        print(f"afrimmlu [{lang}] SKIPPED: {type(e).__name__}: {str(e)[:120]}")

# MMLU STEM (English): deep coverage of high-school + college math and science.
MMLU_STEM = ["high_school_mathematics", "high_school_statistics", "high_school_physics",
             "high_school_chemistry", "high_school_biology", "college_mathematics",
             "college_physics", "college_chemistry", "college_biology", "abstract_algebra",
             "astronomy", "anatomy", "electrical_engineering", "computer_science", "math"]
try:
    cap = TRAIN_CAPS.get("mmlu_stem", 1500)
    n = 0
    for subj in MMLU_STEM:
        try:
            ds = load_dataset("cais/mmlu", subj, split="train")
        except Exception:
            continue
        for row in ds:
            if n >= cap: break
            q = row.get("question"); ans = row.get("answer"); choices = row.get("choices")
            if not q or ans is None or not choices: continue
            cidx = "ABCD".index(ans) if ans in "ABCD" else None
            opt = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices))
            resp = f"The correct option is {ans}." + (f"\n\n{choices[cidx]}" if cidx is not None else "")
            all_pairs.append(to_msgs(f"Answer this question.\n\n{q}\n\n{opt}", resp)); n += 1
        if n >= cap: break
    print(f"mmlu-stem (en): {n}")
except Exception as e:
    print(f"mmlu_stem SKIPPED: {type(e).__name__}: {str(e)[:120]}")

# GSM8K (English): extra English math word problems for deeper arithmetic reasoning.
try:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    cap = TRAIN_CAPS.get("gsm8k_eng", 400)
    n = 0
    for row in ds:
        if n >= cap: break
        all_pairs.append(to_msgs("Solve this mathematics problem step by step.\n\n" + row["question"],
                                 row["answer"]))
        n += 1
    print(f"gsm8k (en): {n}")
except Exception as e:
    print(f"gsm8k SKIPPED: {type(e).__name__}: {str(e)[:120]}")

print("TOTAL training pairs:", len(all_pairs))

# SAFETY: never silently train on a tiny set. If an upstream dataset failed to
# download, all_pairs would be just the curated pairs. Fail loudly so the run
# is not wasted on a meaningless LoRA.
if len(all_pairs) < 60:
    raise RuntimeError(
        f"Only {len(all_pairs)} training pairs built (expected >=60). "
        "An upstream dataset (afrimgsm / afrimmlu / mmlu_stem / gsm8k / waec2007 / hausa_waec) likely failed to "
        "download. Check the SKIPPED messages above, fix access/network, then re-run."
    )
print("OK: training set size is healthy.")

def format_fn(example):
    return tok.apply_chat_template(example["messages"], tokenize=False)

from datasets import Dataset
train_ds = Dataset.from_list([{"messages": m} for m in all_pairs])
print("dataset ready:", len(train_ds))
'''

C_TRAIN = r'''
# Cell 11: QLoRA fine-tune (PEFT + TRL SFTTrainer)
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",   # robust across cohere2 arch
    bias="none", task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=False, fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to=[],
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    formatting_func=format_fn,
    max_seq_length=1024,
)
trainer.train()

# save LoRA adapter
adapter_dir = f"{OUT}/lora_adapter"
model.save_pretrained(adapter_dir)
print("adapter saved to", adapter_dir)
'''

C_POST_LOAD = r'''
# Cell 12: reload base + adapter for post-training eval (same 4-bit setup,
# same eval code = a fair pre/post comparison)
import shutil
del model
gc.collect()
torch.cuda.empty_cache()

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map="auto", torch_dtype=torch.float16)
from peft import PeftModel
model = PeftModel.from_pretrained(model, f"{OUT}/lora_adapter")
model.eval()
print("fine-tuned model loaded")
'''

C_POST_EVAL = r'''
# Cell 13: POST-TRAINING evals (identical harness)
print("== arc_easy (fine-tuned) ==")
ft_arc = eval_arc(model, tok, train_rows, test_rows, n=ARC_N)
print("fine-tuned arc_easy:", ft_arc)
with open(f"{OUT}/ft_arc.json", "w") as f: json.dump(ft_arc, f, indent=2)

print("\n== WAEC bilingual set (fine-tuned) ==")
ft_waec = eval_waec(model, tok, WAEC_EVAL)
by_lang_ft, overall_ft = waec_summary(ft_waec)
print("by language:", by_lang_ft, "| overall:", overall_ft)
with open(f"{OUT}/ft_waec.json", "w") as f:
    json.dump({"by_lang": by_lang_ft, "overall": overall_ft, "items": ft_waec}, f, indent=2, ensure_ascii=False)

print("\n== judge demo (fine-tuned) ==")
ft_demo = {}
for pid, p in DEMO_PROMPTS:
    out = generate_answer(model, tok, chat_prompt(p), max_new=256)
    ft_demo[pid] = out
    print(f"\n--- {pid} ---\n{out}")
with open(f"{OUT}/ft_demo.json", "w") as f:
    json.dump(ft_demo, f, indent=2, ensure_ascii=False)
'''

C_COMPARE = r'''
# Cell 14: comparison + charts + download
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_bar(labels, base_vals, ft_vals, title, fname):
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, base_vals, w, label="base", color="#94a3b8")
    ax.bar(x + w/2, ft_vals, w, label="fine-tuned", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_title(title); ax.legend()
    for xi, b, f in zip(x, base_vals, ft_vals):
        ax.text(xi - w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(xi + w/2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=150); plt.show()

# 1) arc_easy
plot_bar(["arc_easy (loglik)", "arc_easy (gen)"],
         [base_arc["arc_easy_ll_acc"], base_arc["arc_easy_gen_acc"]],
         [ft_arc["arc_easy_ll_acc"], ft_arc["arc_easy_gen_acc"]],
         "arc_easy: base vs fine-tuned (higher is better)", "chart_arc.png")

# 2) WAEC by language
langs = sorted(set([r["lang"] for r in WAEC_EVAL]))
plot_bar(langs,
         [by_lang.get(l, 0) for l in langs],
         [by_lang_ft.get(l, 0) for l in langs],
         "Reasoning set by language: base vs fine-tuned", "chart_waec_lang.png")

# 3) Reasoning overall + judge demo quality marker
plot_bar(["Reasoning overall"],
         [overall], [overall_ft],
         "Reasoning overall accuracy", "chart_waec_overall.png")

# results bundle
summary = {
    "base": {"arc": base_arc, "waec_by_lang": by_lang, "waec_overall": overall},
    "fine_tuned": {"arc": ft_arc, "waec_by_lang": by_lang_ft, "waec_overall": overall_ft},
    "verdict": {
        "arc_improved": ft_arc["arc_easy_ll_acc"] > base_arc["arc_easy_ll_acc"],
        "waec_improved": overall_ft > overall,
    },
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary["verdict"], indent=2))

# download bundle
import shutil
shutil.make_archive("/content/adtc_results", "zip", OUT)
from google.colab import files
files.download("/content/adtc_results.zip")
print("Done. Keep the zip - it contains summary.json + all raw evals + the LoRA adapter.",
)

'''

C_EXPORT = r'''
# Cell 15: merge adapter -> base, convert to GGUF (fp16), quantize Q4_K_M, upload to HF.
# This closes the loop so the submission ships the REAL Hekima model, not base Tiny Aya.
# Heavy step: clones + builds llama.cpp (~2-3 min) and uploads to HF.
import os, subprocess, gc, torch
from huggingface_hub import HfApi, login
from peft import PeftModel

# token must have WRITE scope for upload
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)

# 1) merge LoRA into the float16 base
print("merging LoRA adapter into base (float16)...")
del model; gc.collect(); torch.cuda.empty_cache()
base_fp = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
adapter = PeftModel.from_pretrained(base_fp, f"{OUT}/lora_adapter")
adapter = adapter.merge_and_unload()
MERGED = "/content/hekima_merged"
adapter.save_pretrained(MERGED); tok.save_pretrained(MERGED)
print("merged model ->", MERGED)

# 2) clone + build llama.cpp, then convert + quantize
print("cloning + building llama.cpp (shallow)...")
subprocess.run(["pip", "install", "-q", "cmake"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", "/content/llama.cpp"], check=True)
subprocess.run(["pip", "install", "-q", "-r", "/content/llama.cpp/requirements.txt"], check=True)
subprocess.run(["cmake", "-B", "/content/llama.cpp/build", "-DGGML_CUDA=OFF", "/content/llama.cpp"], check=True)
subprocess.run(["cmake", "--build", "/content/llama.cpp/build", "--config", "Release", "-j", "2"], check=True)
GGUF_FP16 = "/content/hekima_fp16.gguf"
subprocess.run(["python", "/content/llama.cpp/convert_hf_to_gguf.py", MERGED, "--outfile", GGUF_FP16], check=True)
GGUF_Q4 = "/content/hekima-tiny-aya-q4_k_m.gguf"
subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", GGUF_FP16, GGUF_Q4, "Q4_K_M"], check=True)
print("quantized GGUF ->", GGUF_Q4, "| size GB:", round(os.path.getsize(GGUF_Q4) / 1e9, 2))

# 3) upload to HF (optional - requires WRITE token + HF_REPO set)
HF_REPO = os.environ.get("HF_REPO", "coderhema/hekima-tiny-aya-q4_k_m")
try:
    api = HfApi()
    api.create_repo(repo_id=HF_REPO, exist_ok=True)
    api.upload_file(path_or_fileobj=GGUF_Q4, path_in_repo=os.path.basename(GGUF_Q4), repo_id=HF_REPO)
    print("uploaded -> https://huggingface.co/" + HF_REPO)
except Exception as e:
    print("UPLOAD SKIPPED (", type(e).__name__, "):", str(e)[:200])
    print("Local GGUF ready at", GGUF_Q4, "- upload manually or set HF_REPO + WRITE token.")
print("Next: set HEKIMA_MODEL_URL in download_model.sh and update metadata.json model path to this GGUF.")
'''

C_README = r'''
## How to read the results

**What this settles:** whether math and scientific-reasoning fine-tuning (AfriGSM + science + bilingual pairs)
actually helps the automated half of S_acc (arc_easy) and the judge half (reasoning-style
bilingual answers).

- If `arc_easy` improves (verdict `arc_improved: true`): math and scientific-reasoning data aligns with the
  benchmark - the reasoning pivot wins on every axis. Strong line for REPORT.md.
- If `arc_easy` stays flat or drops slightly: not fatal - the +15% African Alpha bonus
  and the qualitative judge half (reasoning bilingual accuracy + step-by-step Yoruba answers)
  are the real pitch. A small arc_easy dip is acceptable if reasoning/language scores rise.
- The absolute arc_easy numbers here are from transformers on Colab (float16, T4).
  The official audit runs llama.cpp + Q4_K_M on the 8 GB laptop, so absolute values
  differ - but the DELTA (base vs fine-tuned) is what matters and transfers.

**Artifacts in the zip:** summary.json (verdict), base/ft arc + waec raw evals,
judge demo transcripts (tp_001 EN + tp_002 YO, before/after), the LoRA adapter,
and the 3 comparison charts.

**Next steps after this run:**
1. Cell 15 (below) merges the adapter -> GGUF Q4_K_M and uploads to HF. After it
   finishes, set `HEKIMA_MODEL_URL` in download_model.sh and update `metadata.json`
   (`model.name` + `_runtime.model_path`) to the uploaded Hekima GGUF.
2. Send me summary.json (or the zip) - I interpret it and update REPORT.md, then we
   run the official profiler (llama.cpp + Q4_K_M on the 8 GB laptop) for S_eff.
'''

# inject data literals into the raw cell sources (before JSON assembly)
def inject(src, placeholder, data):
    # embed as an escaped JSON literal inside json.loads("...")
    escaped = json.dumps(data, ensure_ascii=False).replace("\\", "\\\\").replace('"', '\\"')
    return src.replace(placeholder, escaped)

C_BASE_WAEC = inject(C_BASE_WAEC, "__WAEC_EVAL__", WAEC_EVAL)
C_TRAINDATA = inject(C_TRAINDATA, "__CURATED_TRAIN__", CURATED_TRAIN)

# HF token: read from environment (never hardcode a credential in the repo).
# In Colab: import os; os.environ["HF_TOKEN"] = "HUGGINGFACE_TOKEN_PLACEHOLDER"  (READ scope for the base
# model; WRITE scope if you also run the export Cell 15).
HF_TOKEN = os.environ.get("HF_TOKEN", "")
assert "__HF_TOKEN__" in C_LOGIN, "token placeholder missing"
C_LOGIN = C_LOGIN.replace("__HF_TOKEN__", HF_TOKEN)

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
cells = [
    md("# Hekima - ADTC 2026 Education Fine-Tune Benchmark (base vs QLoRA)\n"
       "\n"
       "**Question:** does fine-tuning tiny-aya-global on math and scientific-reasoning data (AfriGSM math + science "
       "past questions + bilingual step-by-step pairs) improve (a) arc_easy - the automated "
       "half of S_acc - and (b) bilingual reasoning-style answering, the judge half?\n"
       "\n"
       "**Flow:** base evals (arc_easy 50-shot-5 + 20-question WAEC set EN/YO/HA/SW/IG + the 2 "
       "exact metadata.json judge prompts) -> QLoRA fine-tune on T4 -> identical evals -> "
       "comparison charts + verdict JSON. Total runtime ~40-90 min on free Colab T4.\n"
       "\n"
       "Run: Runtime -> Run all. GPU needed (T4 is fine)."),
    md("## Setup\n"
       "1. **HF access (one-time):** open https://huggingface.co/CohereLabs/tiny-aya-global, "
       "click *Agree and access repository* (instant).\n"
       "2. **HF token (one-time):** https://huggingface.co/settings/tokens -> create a *read* "
       "token.\n"
       "3. Run Cell 1 (install), Cell 2 (login, paste token), then continue."),
    code(C_INSTALL),
    code(C_LOGIN),
    code(C_CONFIG),
    code(C_LOAD),
    md("## Baseline (before fine-tuning)"),
    code(C_EVALHELP),
    code(C_ARCDATA),
    code(C_BASE_EVAL),
    code(C_BASE_WAEC),
    code(C_BASE_DEMO),
    md(C_TRAIN_HEADER),
    code(C_TRAINDATA),
    code(C_TRAIN),
    md("## Post-training (same harness)"),
    code(C_POST_LOAD),
    code(C_POST_EVAL),
    code(C_COMPARE),
    code(C_EXPORT),
    md(C_README),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = os.path.join(HERE, "adtc_qlora_benchmark.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# also persist the data files for the repo
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "reasoning_eval_questions.json"), "w", encoding="utf-8") as f:
    json.dump({"description": "Bilingual reasoning MCQ eval set (EN/YO/HA/SW/IG), used in the Colab benchmark notebook. Answers verified by construction (simple math).",
               "questions": WAEC_EVAL}, f, indent=2, ensure_ascii=False)
with open(os.path.join(DATA, "curated_train_pairs.json"), "w", encoding="utf-8") as f:
    json.dump({"description": "Curated step-by-step training pairs (EN + YO + IG + HA + SW) with LaTeX math formatting. All non-English pairs need native-speaker validation.",
               "pairs": CURATED_TRAIN}, f, indent=2, ensure_ascii=False)

print("wrote", out_path)
print("wrote", os.path.join(DATA, "reasoning_eval_questions.json"))
print("wrote", os.path.join(DATA, "curated_train_pairs.json"))
