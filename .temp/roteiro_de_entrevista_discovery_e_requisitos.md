# Roteiro de Entrevista para Discovery e Levantamento de Requisitos

> **Objetivo:** Identificar a causa-raiz do problema de negócio, validar a viabilidade técnica/dados e alinhar expectativas de entrega antes de propor qualquer ferramenta, arquitetura ou modelo.

---

## 1. Diagnóstico do Problema e Impacto de Negócio
*O foco desta etapa é desarmar soluções pré-concebidas trazidas pelo cliente e entender a dor real.*

1. **Definição da Dor:** Em termos práticos e simples, qual é o problema ou gargalo central que motivou esta conversa?
2. **Impacto Financeiro/Operacional:** O que a organização perde hoje (tempo de equipe, receita direta, retenção de clientes) por não ter essa questão resolvida?
3. **Decisão Acionável:** Qual decisão específica de negócio precisa ser tomada ou melhorada com os resultados que vamos gerar?
4. **Tentativas Anteriores:** Essa iniciativa já foi tentada antes internamente ou com parceiros externos? Se sim, quais foram os principais motivos para não ter avançado?

---

## 2. Métricas de Sucesso e Baseline
*O objetivo é estabelecer critérios quantificáveis para definir claramente o que significa sucesso no projeto.*

5. **Métrica Atual (Baseline):** Como esse problema é medido atualmente? Qual é o indicador numérico de partida (ex.: horas gastas/mês, taxa de conversão, tempo médio de resolução)?
6. **Critério de Sucesso (KPI):** Qual resultado quantitativo tornará este projeto indiscutivelmente bem-sucedido para a liderança em um horizonte de 3 a 6 meses?
7. **Tolerância e Custo do Erro:** Quando a solução errar (ex.: um falso positivo, uma previsão imprecisa ou uma falha de classificação), qual é a consequência prática na operação?

---

## 3. Mapeamento do Processo Atual (As-Is)
*Mapear o fluxo operacional ponta a ponta para evitar a automação de processos ineficientes.*

8. **Fluxo Ponta a Ponta:** Qual é o caminho que a informação ou tarefa percorre hoje, desde o evento inicial até a conclusão?
9. **Gargalos Operacionais:** Em qual ponto do fluxo ocorrem os maiores atritos, atrasos manuais ou retrabalhos frequentes?
10. **Stakeholders e Usuários:**
    - Quem executa a rotina no dia a dia?
    - Quem consome o produto final?
    - Quem é o decisor responsável pela aprovação da entrega?

---

## 4. Maturidade e Viabilidade de Dados / Infraestrutura
*Garantir a existência das condições técnicas necessárias antes do início do desenvolvimento.*

11. **Fontes de Dados:** Onde residem as informações necessárias para resolver o problema (bancos relacionais, planilhas descentralizadas, logs, APIs legadas, ERP/CRM)?
12. **Histórico e Atualização:** Quanto tempo de histórico confiável está disponível e com que periodicidade os dados são atualizados?
13. **Qualidade dos Dados:** Existem lacunas conhecidas na base, como campos essenciais em branco, duplicidades ou falta de padronização cadastral?
14. **Ambiente Tecnológico:** Onde a solução precisará rodar (nuvem do cliente, ambiente local/on-premise, container, integração direta via webhook/API)?

---

## 5. Forma de Entrega, Governança e Restrições
*Garantir que a entrega final se encaixe na rotina real dos usuários e cumpra as restrições da empresa.*

15. **Interface de Consumo:** Como a equipe prefere interagir com a solução (painel/dashboard analítico, alerta automatizado em ferramenta de comunicação, endpoint de API ou relatório consolidado)?
16. **Janela de Entrega:** Existe algum prazo crítico ou marco institucional inegociável para a primeira versão em produção?
17. **Compliance e Segurança:** Quais exigências de governança, LGPD/privacidade ou políticas de segurança interna devem ser observadas?
18. **Ganhos Rápidos (*Quick Wins*):** Existe alguma entrega menor e de alto impacto que podemos disponibilizar nas primeiras semanas enquanto a solução definitiva é construída?