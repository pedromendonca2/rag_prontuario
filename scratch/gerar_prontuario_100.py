#!/usr/bin/env python3
"""
Gera um prontuário sintético de 100 evoluções cronológicas realistas
emulando os padrões clínicos brasileiros de PS e Enfermaria/Ambulatório.
Indexa no ChromaDB sob o paciente_id: 'paciente_100_evol'.
"""
import os
import json
import random
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "/app/chroma_db" if os.path.exists("/app/chroma_db") else "/home/ease/chroma_db"
DINAMICOS_PATH = "/app/prontuarios_dinamicos.json" if os.path.exists("/app") else "/home/ease/rag_prontuario/prontuarios_dinamicos.json"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PACIENTE_ID = "paciente_100_evol"
NOME_PACIENTE = "Paciente Modelo (100 Evoluções)"

def gerar_100_evolucoes():
    data_inicio = datetime(2024, 1, 15)
    dias_acumulados = 0

    evolucoes = []

    # Linha do tempo clínica com eventos variados
    tipos = [
        "Consulta Ambulatorial", "Retorno Clínico", "Pronto Atendimento", 
        "Exame Complementar", "Intercorrência", "Reavaliação Médica"
    ]

    # Condições clínicas ao longo dos 2 anos
    medicacoes = [
        "Losartana 50mg/dia", "Losartana 100mg/dia", "Metformina 850mg 2x/dia",
        "Anlodipino 5mg/dia", "HCTZ 25mg/dia", "Sinvastatina 20mg/dia",
        "Budesonida 200mcg spray 12/12h", "Salbutamol spray resgate"
    ]

    for i in range(1, 101):
        dias_acumulados += random.randint(7, 12)
        data_atual = data_inicio + timedelta(days=dias_acumulados)
        data_str = data_atual.strftime("%d/%m/%Y")

        tipo = random.choice(tipos)
        pa_sis = random.randint(118, 155)
        pa_dia = random.randint(72, 96)
        fc = random.randint(64, 88)
        sat = random.randint(96, 99)
        tax = round(random.uniform(36.0, 36.6), 1)

        if i == 1:
            tipo = "Consulta Inicial"
            texto = f"""### CONSULTA AMBULATORIAL INICIAL ###
# HDA: Paciente masculino, 58 anos, comparece para avaliação e acompanhamento clínico. Refere cefaleia holocraniana e episódios de mal-estar há 2 meses. Nega febre, dor torácica ou dispneia no momento.
# HPP:
- Comorbidades: Hipertensão Arterial Sistêmica (HAS) e Diabetes Mellitus tipo 2 (DM2) diagnosticados há 3 anos com adesão irregular.
- Alergias medicamentosas: Nega.
- Medicações em uso: Losartana 50mg/dia, Metformina 850mg no almoço.
# SINAIS VITAIS: PA: {pa_sis}/{pa_dia} mmHg | FC: {fc} bpm | Sat: {sat}% | Temp: {tax}ºC
# EXAME FÍSICO: BEG, LOTE, corado, hidratado, anictérico, acianótico. ACV: RCR 2T, BNF s/ sopros. AR: MVF s/ ruídos adventícios. Abdome flácido, indolor à palpação, RHA+. MMII sem edemas.
# HD: HAS descompensada / DM2 a esclarecer controle.
# CONDUTA: Ajusto Losartana para 50mg 12/12h, mantenho Metformina 850mg 2x/dia. Solicito exames laboratoriais de rotina e ECG. Retorno em 30 dias."""
        elif i % 10 == 0:
            tipo = "Pronto Atendimento"
            texto = f"""### ATENDIMENTO DE URGÊNCIA (PS) ###
# HDA: Paciente da entrada com queixa de pico hipertensivo associado a cefaleia occipital de forte intensidade e palpitações de início há 4 horas após estresse emocional. Nega déficit focal motor ou dor precordial em aperto.
# SINAIS VITAIS: PA: 172/104 mmHg | FC: 96 bpm | Sat: 97% | Temp: 36.2ºC
# EXAME FÍSICO: REG, LOTE, orientado. Neuro: Glasgow 15, pupilas isocóricas e fotorreagentes, sem sinais meníngeos. ACV: RCR 2T BNF. AR: MV presente bilateral sem sibilos.
# CONDUTA: Captopril 25mg VO via sublingual + Dipirona 1g EV. Realizado ECG (sem sinais de isquemia aguda).
## REAVALIAÇÃO APÓS 1H: Paciente refere remissão completa da cefaleia. PA aferida em 134/82 mmHg, FC 78 bpm. Alta com orientações e encaminhamento para manter conduta com médico assistente."""
        elif i % 15 == 0:
            tipo = "Exame Complementar"
            glic = random.randint(105, 145)
            hba1c = round(random.uniform(6.8, 7.6), 1)
            cr = round(random.uniform(1.0, 1.25), 2)
            ur = random.randint(32, 48)
            col_tot = random.randint(180, 215)
            ldl = random.randint(105, 138)
            hdl = random.randint(42, 54)
            trig = random.randint(130, 175)
            texto = f"""### RESULTADO DE EXAMES LABORATORIAIS E COMPLEMENTARES ###
- Hemograma: Hemácias 4,92 | Hb 14,3 g/dL | Ht 42,8% | Leucócitos 6.400 (segmentados 62%, linfócitos 28%) | Plaquetas 242.000.
- Perfil Glicêmico e Renal: Glicemia em jejum {glic} mg/dL | HbA1c {hba1c}% | Creatinina {cr} mg/dL | Ureia {ur} mg/dL | Na 140 mEq/L | K 4,2 mEq/L.
- Perfil Lipídico: Colesterol Total {col_tot} mg/dL | LDL {ldl} mg/dL | HDL {hdl} mg/dL | Triglicerídeos {trig} mg/dL.
- EAS: Densidade 1.018 | pH 6.0 | Proteínas ausentes | Glicose ausente | Leucócitos 2/campo | Hemácias 1/campo | Ausência de piócitos ou nitrito.
- Laudo ECG: Ritmo sinusal regular, FC 72 bpm, eixo elétrico preservado, sem alterações isquêmicas de repolarização ventricular."""
        else:
            tipo = "Retorno Ambulatorial"
            queixa = random.choice([
                "Paciente assintomático, refere boa tolerância medicamentosa e controle dietético.",
                "Refere episódios ocasionais de tosse seca noturna, nega dor no peito ou edema.",
                "Traz anotações domiciliares de PA com médias de 125x80 mmHg. Satisfeito com evolução.",
                "Refere melhora substancial da disposição e caminhadas 3 vezes por semana.",
                "Refere leve episódio de epigastralgia após refeição gordurosa, em melhora espontânea."
            ])
            conduta = random.choice([
                "Mantida prescrição atual de Losartana e Metformina. Reforçadas medidas de estilo de vida.",
                "Mantidas medicações. Solicitada ultrassonografia de controle e retorno em 60 dias.",
                "Boa adesão terapêutica evidenciada. Agendado novo retorno semestral.",
                "Prescrito reposição vitamínica e orientado controle rigoroso da glicemia matinal.",
                "Conduta clínica mantida. Orientado retorno imediato em caso de sinais de alarme."
            ])
            texto = f"""### ATENDIMENTO CLÍNICO - RETORNO #{i:02d} ###
# DATA: {data_str}
# HDA: {queixa}
# SINAIS VITAIS: PA: {pa_sis}/{pa_dia} mmHg | FC: {fc} bpm | Sat: {sat}% | Temp: {tax}ºC
# EXAME FÍSICO: BEG, LOTE, mucosas coradas e hidratadas. ACV: RCR 2T, BNF s/ sopros. AR: MVF sem ruídos adventícios. Abdome flácido, indolor, sem visceromegalias. Extremidades bem perfundidas, sem edemas.
# CONDUTA: {conduta}"""

        evolucoes.append({
            "id": f"{PACIENTE_ID}_{i:03d}",
            "paciente_id": PACIENTE_ID,
            "nome": NOME_PACIENTE,
            "categoria": "Acompanhamento Crônico",
            "data": data_str,
            "tipo_evento": tipo,
            "profissional": "Dr. Clínico Assistente",
            "texto": texto.strip()
        })

    return evolucoes

def main():
    print("=" * 65)
    print(f">>> Gerando 100 evoluções sintéticas para o paciente: {PACIENTE_ID}...")
    print("=" * 65)

    novas_100 = gerar_100_evolucoes()

    # 1. Carrega prontuários dinâmicos existentes (os 20 reais sanitizados)
    registros_existentes = []
    if os.path.exists(DINAMICOS_PATH):
        try:
            with open(DINAMICOS_PATH, "r", encoding="utf-8") as f:
                registros_existentes = json.load(f)
        except Exception:
            registros_existentes = []

    # Remove qualquer versão anterior do paciente_100_evol se houver
    registros_filtrados = [r for r in registros_existentes if r.get("paciente_id") != PACIENTE_ID]
    registros_totais = registros_filtrados + novas_100

    with open(DINAMICOS_PATH, "w", encoding="utf-8") as f:
        json.dump(registros_totais, f, ensure_ascii=False, indent=2)
    print(f"[OK] {DINAMICOS_PATH} atualizado com {len(novas_100)} novas evoluções.")

    # 2. Indexa no ChromaDB
    print("\nConectando ao ChromaDB e indexando os 100 registros...")
    cliente = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    colecao = cliente.get_collection("prontuario_evolucoes")

    # Deleta registros antigos desse paciente_id se existirem
    try:
        colecao.delete(where={"paciente_id": PACIENTE_ID})
    except Exception:
        pass

    modelo = SentenceTransformer(MODELO_EMBEDDING)

    ids = [r["id"] for r in novas_100]
    textos = [r["texto"] for r in novas_100]
    metadatas = [
        {
            "paciente_id": PACIENTE_ID,
            "nome": NOME_PACIENTE,
            "categoria": r["categoria"],
            "data": r["data"],
            "tipo_evento": r["tipo_evento"],
            "profissional": r["profissional"]
        }
        for r in novas_100
    ]

    print("Calculando embeddings em lote...")
    batch_size = 25
    for b in range(0, len(ids), batch_size):
        fim = min(b + batch_size, len(ids))
        sub_ids = ids[b:fim]
        sub_textos = textos[b:fim]
        sub_metas = metadatas[b:fim]
        sub_embeddings = modelo.encode(sub_textos, show_progress_bar=False).tolist()

        colecao.add(
            ids=sub_ids,
            embeddings=sub_embeddings,
            metadatas=sub_metas,
            documents=sub_textos
        )
        print(f"  -> Indexados registros {b+1} até {fim} de 100")

    print("\n" + "=" * 65)
    print(f"🎉 SUCESSO! Paciente '{PACIENTE_ID}' criado com exatamente 100 evoluções.")
    print(f"Total de documentos agora no ChromaDB: {colecao.count()}")
    print("=" * 65)

if __name__ == "__main__":
    main()

