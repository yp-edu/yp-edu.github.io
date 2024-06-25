
(function(document) {
    var bloquotes = document.getElementsByTagName('blockquote');
    for (var i = 0; i < bloquotes.length; i++) {
        var first_p = bloquotes[i].getElementsByTagName('p')[0];
        var p_html = first_p.innerHTML;
        matches = p_html.match(/\[\![a-z]{1,}\]/g);
        if (matches) {
            let callout = matches[0].slice(2, -1);
            console.log(callout);
            var div_title = document.createElement('div');
            var span = document.createElement('em');
            span.innerHTML = p_html.replace(/\[\![a-z]{1,}\]/g, '');

            div_title.classList.add('callout-title');
            div_title.innerHTML = '<i class="fa-solid fa-fire-flame-curved" href="#"></i>'
            div_title.appendChild(span);
            
            first_p.parentNode.replaceChild(div_title, first_p);
            bloquotes[i].classList.add('callout');
            bloquotes[i].classList.add(callout);
        }

    }
})(document);