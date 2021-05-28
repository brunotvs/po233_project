# PO-233 - 01/2021

Ainda estou testando a parte do setup.py, os dados do projeta são difíceis de baixar.

A estrutura está como:
paper
arquivos relativos ao artigo - ainda sem estrutura
source
build_data - pacote python para baixar os dados do projeta e construir um banco estruturado - ainda não funciona
model - modelo de machine learning

como ainda não estamos usando um ambiente python, acredito que podem ignorar os arquivos setup.py e requirements.txt

Tuto de git:
Após fazer as edições e salvar os arquivos que está trabalhando:
git add "caminho relativo dos arquivos que quer fazer download" ou git add -A para adicionar todos
exemplo:
git add requirements.txt setup.py
git add paper/introdução.tex

    Após adicionar os arquivos que gostaria de adicionar à versão:
        git commit -m "Comentário das edições que fez"
        exemplo:
            git commit -m "Citei o artigo de fulano na introdução"
            git commit -m "implementei um algorítmo novo no código"

        essa parte que vai criar uma nova versão dos arquivos editados e salvar as versões anteriores como histórico

    Após isso:
        para baixar as atualizações que estão no servidor do github:
            git pull
        para fazer upload das versões que você está trabalhando:
            git push
Softwares para baixar:
    Recomendados:
        Visual studio code [https://code.visualstudio.com/] - alternativa seria qualquer editor, como vim, sublime etc.
        Tex Live [https://www.tug.org/texlive/] - alternativa seria mik tex, mas nunca usei e não sei se funciona bem
    "Obrigatórios":
        Inkscape [https://inkscape.org/pt-br/] - necessário para utilizar svg no documento (precisa adicionar à variável path do sistema)
        git-scm [https://git-scm.com/] - utilizar o git
        R e python, para códigos

<<<<<<< HEAD
=======
    Teste
>>>>>>> 3ec39d3abe9d085fb7e1a9d113beafd16027bad8
