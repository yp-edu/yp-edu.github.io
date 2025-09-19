// Change the visibility of the navigation bar when clicking on #menu-icon

(function(document) {
    var menu_icon = document.getElementById('menu-icon')
    menu_icon.addEventListener('click', function(e) {
        e.preventDefault();
        var nav = document.getElementById('menu');
        if (nav.style.visibility == 'hidden') {
            nav.style.visibility = 'visible';
            menu_icon.classList.add('menu-active');
        }
        else {
            nav.style.visibility = 'hidden';
            menu_icon.classList.remove('menu-active');
        }
        
    }, false);
})(document);