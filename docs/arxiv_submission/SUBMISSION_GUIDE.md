# Submission Guide

## arXiv (fazer primeiro — imediato)

1. Acesse https://arxiv.org/submit
2. Categoria primária: **cs.PL** (Programming Languages)
   Cross-list: **cs.SE** (Software Engineering)
3. Faça upload de `main.tex` (o arquivo .bbl não é necessário pois as referências estão inline no `\thebibliography`)
4. Title: `Bridging the Ergonomics Gap in Python JIT Compilation: AST-Level Source Transformation for Numba`
5. Authors: `Rodrigo Martinez Pinto`
6. Abstract: copiar o texto do `\begin{abstract}` no main.tex
7. License: **CC BY 4.0** (recomendado para máxima indexação)

**Nota sobre endorsement:** Primeira submissão em cs.PL pode exigir endorsement.
Se solicitado, use o email institucional (contact@cloudsealed.com) ou peça
endorsement a um pesquisador da área em https://arxiv.org/auth/endorse

---

## IEEE Software Magazine (depois do arXiv)

- URL: https://www.computer.org/csdl/magazine/so
- Track: **"Software Engineering in Practice"** (Feature Article)
- Sistema: ScholarOne Manuscripts
- Limite: 4.200 palavras (figuras/tabelas contam 250 cada)
- Referências: máximo 15
- Abstract: máximo 150 palavras

### Checklist antes de submeter

- [ ] Converter main.tex para o template oficial IEEE Software
      (baixar em https://www.ieee.org/conferences/publishing/templates.html)
- [ ] Verificar contagem de palavras ≤ 4.200
- [ ] Verificar abstract ≤ 150 palavras
- [ ] Verificar referências ≤ 15
- [ ] Incluir author bio (já está no main.tex)
- [ ] Adicionar o link do arXiv preprint na cover letter

### Cover letter sugerida

> Dear IEEE Software Editorial Board,
>
> I submit for your consideration the manuscript "Bridging the Ergonomics Gap
> in Python JIT Compilation: AST-Level Source Transformation for Numba" for
> the Software Engineering in Practice section.
>
> This work addresses a concrete usability barrier in Numba, a widely adopted
> Python JIT compiler: its nopython mode rejects f-strings, list comprehensions,
> keyword arguments, and PEP 484 type annotations — all idiomatic in modern
> Python. The CloudSealed Compiler resolves these limitations through two
> AST-level source transformations, a type mapper, and a structured diagnostic
> layer, without modifying Numba itself and without requiring a build step.
>
> The work is: (1) practically motivated — the Numba issue tracker has
> hundreds of reports of these limitations; (2) technically novel — no existing
> tool addresses this specific gap; (3) evaluated — with a 38-case test suite
> and measured benchmarks; (4) reproducible — the open-source package is
> available at pip install cloudsealed-jit.
>
> A preprint is available at [arXiv link after submission].
>
> Thank you for your consideration.
> Rodrigo Martinez Pinto

---

## CGO 2027 Tool Paper (alternativa de alto impacto)

- URL: https://2027.cgo.org (deadlines tipicamente setembro/outubro)
- Formato: IEEE conference, 10 páginas, referências não contam
- Requisito: Artifact Evaluation obrigatório para Tool Papers
- O projeto já atende os critérios de CGO:
  - Originality: AST-level bridge for Numba not done before
  - Usability: pip install, MIT license, public GitHub
  - Completeness: 38 tests, benchmarks, documented API
