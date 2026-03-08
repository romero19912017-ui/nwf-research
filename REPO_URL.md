# URL репозитория GitHub

Чтобы указать ваш новый репозиторий, замените во всех файлах:

```
https://github.com/YOUR_USERNAME/nwf-research
```

на ваш URL, например:

```
https://github.com/username/nwf-research
```

**Файлы с ссылкой на репозиторий:**
- README.md
- OTCHET_EKSPERIMENTOV.md
- OTCHET_DETALNYY_SRAVNENIE.md
- HABR_ARTICLE_DRAFT.md

**Команда для замены (PowerShell):**
```powershell
$url = "https://github.com/ВАШ_USERNAME/nwf-research"
Get-ChildItem -Recurse -Include *.md | ForEach-Object {
  (Get-Content $_.FullName) -replace 'https://github.com/YOUR_USERNAME/nwf-research', $url | Set-Content $_.FullName
}
```
